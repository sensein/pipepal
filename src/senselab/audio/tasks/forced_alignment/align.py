"""Align function based on WhisperX implementation."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.forced_alignment.constants import (
    LANGUAGES_WITHOUT_SPACES,
    PUNKT_ABBREVIATIONS,
)
from senselab.audio.tasks.forced_alignment.data_structures import (
    AlignedTranscriptionResult,
    Point,
    Segment,
    SingleAlignedSegment,
    SingleSegment,
    SingleWordSegment,
)
from senselab.utils.data_structures.script_line import ScriptLine


def _prepare_audio(audio: Audio) -> Audio:
    """Prepare audio data for processing.

    Args:
        audio (Union[np.ndarray, torch.Tensor]): The audio data to be prepared.

    Returns:
        torch.Tensor: The prepared audio data as a torch tensor.
    """
    if not torch.is_tensor(audio.waveform):
        audio.waveform = torch.from_numpy(audio.waveform)
    if len(audio.waveform.shape) == 1:
        audio.waveform = audio.waveform.unsqueeze(0)
    return audio


def _preprocess_segments(
    transcript: List[SingleSegment],
    model_dictionary: Dict[str, int],
    model_lang: str,
    print_progress: bool,
    combined_progress: bool,
) -> List[SingleSegment]:
    """Preprocess transcription segments by filtering characters, handling spaces, and preparing text.

    Args:
        transcript (List[SingleSegment]): The list of transcription segments.
        model_dictionary (Dict[str, int]): Dictionary for the alignment model.
        model_lang (str): Language of the model.
        print_progress (bool): Whether to print progress.
        combined_progress (bool): Whether to combine progress percentage.

    Returns:
        List[SingleSegment]: The preprocessed transcription segments.
    """
    total_segments = len(transcript)

    for sdx, segment in enumerate(transcript):
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # Split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = [text]

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                continue
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any(c in model_dictionary.keys() for c in wrd):
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans

    return transcript


def _can_align_segment(
    segment: SingleSegment, model_dictionary: Dict[str, int], t1: float, max_duration: float
) -> bool:
    """Checks if a segment can be aligned.

    Args:
        segment (SingleSegment): The segment to check.
        model_dictionary (Dict[str, int]): Dictionary for character indices.
        t1 (float): Start time of the segment.
        max_duration (float): Maximum duration of the audio.

    Returns:
        bool: True if the segment can be aligned, False otherwise.
    """
    if segment["clean_char"] is None or len(segment["clean_char"]) == 0:
        return False
    if t1 >= max_duration:
        return False
    return True


def _prepare_waveform_segment(
    audio: Audio, t1: float, t2: float, device: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Prepares the waveform segment based on the time points.

    Args:
        audio (Audio): The audio data.
        t1 (float): Start time of the segment.
        t2 (float): End time of the segment.
        device (str): The device to run the model on.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: The waveform segment and its length.
    """
    f1 = int(t1 * audio.sampling_rate)
    f2 = int(t2 * audio.sampling_rate)

    waveform_segment = audio.waveform[:, f1:f2]
    if isinstance(waveform_segment, np.ndarray):
        waveform_segment = torch.from_numpy(waveform_segment)

    if waveform_segment.shape[-1] < 400:
        lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
        waveform_segment = torch.nn.functional.pad(waveform_segment, (0, 400 - waveform_segment.shape[-1]))
    else:
        lengths = None

    return waveform_segment, lengths


def _get_prediction_matrix(
    model: torch.nn.Module,
    waveform_segment: torch.Tensor,
    lengths: Optional[torch.Tensor],
    model_type: str,
    device: str,
) -> torch.Tensor:
    """Generate prediction matrix from the alignment model.

    Args:
        model (torch.nn.Module): The alignment model.
        waveform_segment (torch.Tensor): The audio segment to be processed.
        lengths (Optional[torch.Tensor]): Lengths of the audio segments.
        model_type (str): The type of the model ('torchaudio' or 'huggingface').
        device (str): The device to run the model on.

    Returns:
        torch.Tensor: The prediction matrix.
    """
    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model(waveform_segment.to(device), lengths=lengths)
        elif model_type == "huggingface":
            emissions = model(waveform_segment.to(device)).logits
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")

        emissions = torch.log_softmax(emissions, dim=-1)

    return emissions


def _get_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
    """Gets the trellis for token alignment.

    Args:
        emission (torch.Tensor): The emission matrix from the model.
        tokens (List[int]): The token IDs.
        blank_id (int): The ID for the blank token.

    Returns:
        torch.Tensor: The trellis matrix.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def _backtrack(
    trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int = 0
) -> Optional[List[Point]]:
    """Backtracks to find the best path through the trellis.

    Args:
        trellis (torch.Tensor): The trellis matrix.
        emission (torch.Tensor): The emission matrix from the model.
        tokens (List[int]): The token IDs.
        blank_id (int): The ID for the blank token.

    Returns:
        Optional[List[Point]]: The best path as a list of Points.
    """
    j = trellis.size(1) - 1
    t_start = int(torch.argmax(trellis[:, j]).item())

    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        path.append(Point(j - 1, t - 1, prob))

        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


def _merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    """Merges repeated tokens in the alignment path.

    Args:
        path (List[Point]): The alignment path.
        transcript (str): The transcript text.

    Returns:
        List[Segment]: The merged segments.
    """
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def _interpolate_nans(x: pd.Series, method: str = "nearest") -> pd.Series:
    """Interpolates NaN values in a pandas Series.

    Args:
        x (pd.Series): The pandas Series.
        method (str): The interpolation method (default: "nearest").

    Returns:
        pd.Series: The Series with interpolated NaNs.
    """
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()


def _align_segments(
    transcript: List[SingleSegment],
    model: torch.nn.Module,
    model_dictionary: Dict[str, int],
    model_lang: str,
    model_type: str,
    audio: torch.Tensor,
    device: str,
    max_duration: float,
    return_char_alignments: bool,
    interpolate_method: str,
) -> tuple[list[SingleAlignedSegment], list[SingleWordSegment]]:
    """Align segments based on the predictions.

    Args:
        transcript (List[SingleSegment]): The list of transcription segments.
        model (torch.nn.Module): The alignment model.
        model_dictionary (Dict[str, int]): Dictionary for character indices.
        model_lang (str): Language of the model.
        model_type (str): The type of the model ('torchaudio' or 'huggingface').
        audio (torch.Tensor): The audio data.
        device (str): The device to run the model on.
        max_duration (float): Maximum duration of the audio.
        return_char_alignments (bool): Flag to return character alignments.
        interpolate_method (str): Method for interpolating NaNs.

    Returns:
        List[SingleAlignedSegment]: The aligned segments.
    """
    aligned_segments = []
    word_segments = []

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}

        if return_char_alignments:
            aligned_seg["chars"] = []

        # Check if we can align
        if not _can_align_segment(segment, model_dictionary, t1, max_duration):
            print(f'Failed to align segment ("{segment["text"]}"), skipping...')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment["clean_char"] or [])
        tokens = [model_dictionary[c] for c in text_clean]

        waveform_segment, lengths = _prepare_waveform_segment(audio, t1, t2, device)

        emissions = _get_prediction_matrix(model, waveform_segment, lengths, model_type, device)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == "[pad]" or char == "<pad>":
                blank_id = code

        trellis = _get_trellis(emission, tokens, blank_id)
        path = _backtrack(trellis, emission, tokens, blank_id)

        if path is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        char_segments = _merge_repeats(path, text_clean)

        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # Assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if segment["clean_cdx"] is not None and cdx in segment["clean_cdx"]:
                char_seg = char_segments[segment["clean_cdx"].index(cdx)]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                word_idx += 1

        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        if isinstance(char_segments_arr, pd.DataFrame):
            char_segments_arr["sentence-idx"] = None
        else:
            raise TypeError("char_segments_arr must be a pandas DataFrame.")

        if segment["sentence_spans"] is not None:
            for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
                curr_chars = char_segments_arr.loc[
                    (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)
                ]
                char_segments_arr.loc[
                    (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"
                ] = sdx

                sentence_text = text[sstart:send]
                sentence_start = curr_chars["start"].min()
                end_chars = curr_chars[curr_chars["char"] != " "]
                sentence_end = end_chars["end"].max()
                sentence_words = []

                for word_idx in curr_chars["word-idx"].unique():
                    word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                    word_text = "".join(word_chars["char"].tolist()).strip()
                    if len(word_text) == 0:
                        continue

                    word_chars = word_chars[word_chars["char"] != " "]

                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

                    word_segment = SingleWordSegment(word=word_text, start=word_start, end=word_end, score=word_score)

                    sentence_words.append(word_segment)
                    word_segments.append(word_segment)

                aligned_subsegment = SingleAlignedSegment(
                    text=sentence_text, start=sentence_start, end=sentence_end, words=sentence_words, chars=word_chars
                )
                aligned_subsegments.append(aligned_subsegment)

                if return_char_alignments:
                    curr_chars = curr_chars[["char", "start", "end", "score"]]
                    curr_chars.fillna(-1, inplace=True)
                    curr_chars = curr_chars.to_dict("records")
                    curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                    aligned_subsegments[-1]["chars"] = curr_chars

            if aligned_subsegments:
                aligned_subsegments_df = pd.DataFrame(aligned_subsegments)

                aligned_subsegments_df["start"] = _interpolate_nans(
                    aligned_subsegments_df["start"], method=interpolate_method
                )
                aligned_subsegments_df["end"] = _interpolate_nans(
                    aligned_subsegments_df["end"], method=interpolate_method
                )
                agg_dict = {"text": " ".join, "words": "sum"}
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    agg_dict["text"] = "".join
                if return_char_alignments:
                    agg_dict["chars"] = "sum"
                aligned_subsegments_df.groupby(["start", "end"], as_index=False).agg(agg_dict)
                aligned_subsegments = aligned_subsegments_df.to_dict("records")

        aligned_segments.extend(aligned_subsegments)

    return (aligned_segments, word_segments)


def convert_to_scriptline(data: AlignedTranscriptionResult) -> List[ScriptLine]:
    """Convert a dictionary of segments and word segments to a list of ScriptLine objects.

    Args:
        data (AlignedTranscriptionResult): The input dictionary with segments and word segments.

    Returns:
        List[ScriptLine]: The list of ScriptLine objects.
    """
    segments = data["segments"]
    script_lines = []

    for segment in segments:
        words = segment["words"]
        word_chunks = [ScriptLine(text=word["word"]) for word in words]

        # Handle 'nan' end values by setting them to None
        start = segment["start"]
        end: Optional[float] = segment["end"]
        if end is not None and (isinstance(end, float) and math.isnan(end)):
            end = None

        script_line = ScriptLine(text=segment["text"], start=start, end=end, chunks=word_chunks)
        script_lines.append(script_line)

    return script_lines


def align(
    transcript: List[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: Dict[str, Any],
    audio: Audio,
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """Aligns phoneme recognition predictions to known transcription.

    Args:
        transcript (List[SingleSegment]): The list of transcription segments.
        model (torch.nn.Module): The alignment model.
        align_model_metadata (Dict[str, Any]): Metadata for the alignment model.
        audio (np.ndarray): The audio data.
        device (str): The device to run the model on.
        interpolate_method (str): The method for interpolating NaNs (default: "nearest").
        return_char_alignments (bool): Whether to return character alignments (default: False).
        print_progress (bool): Whether to print progress (default: False).
        combined_progress (bool): Whether to combine progress (default: False).

    Returns:
        AlignedTranscriptionResult: The aligned transcription result.
    """
    audio = _prepare_audio(audio)
    max_duration = audio.waveform.shape[1] / audio.sampling_rate

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    transcript = _preprocess_segments(
        transcript,
        align_model_metadata["dictionary"],
        align_model_metadata["language"],
        print_progress,
        combined_progress,
    )

    aligned_segments, word_segments = _align_segments(
        transcript=transcript,
        model=model,
        model_dictionary=model_dictionary,
        model_lang=model_lang,
        model_type=model_type,
        audio=audio,
        device=device,
        max_duration=max_duration,
        return_char_alignments=return_char_alignments,
        interpolate_method=interpolate_method,
    )

    return {"segments": aligned_segments, "word_segments": word_segments}


# Note: most of this code is from: https://github.com/m-bain/whisperX/tree/main

# Copyright (c) 2022, Max Bain
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by Max Bain.
# 4. Neither the name of Max Bain nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.