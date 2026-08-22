from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Structured arm: measure-then-decide.
#
# The model sees only the image -- never a mask -- and is asked to estimate
# visual proxies for the same quantities datset_creation/mask_metrics.py
# computes from the expert mask, using the SAME thresholds. That is what makes
# the run analysable two ways at once: label agreement against the mask-derived
# `lesion_*_label` columns, and numeric agreement against the mask-derived
# measurement columns (border_solidity, echo_lesion_to_reference_ratio, ...).
#
# The thresholds below are therefore deliberately NOT tuned. Changing one here
# breaks comparability with the ground-truth columns; ablate the decision stage
# instead, which is separated out per task on purpose.
#
# Differences from the seven single-feature prompts this replaces:
#   * one pass instead of seven, so the arm costs one generation per image;
#   * the posterior rule is made explicit -- "shadowing if shadowing dominates"
#     was undefined, and is now the same global-ratio test mask_metrics uses;
#   * null handling is specified, so an unmeasurable feature degrades the score
#     rather than aborting the decision;
#   * a decision stage is appended, which the originals stopped short of.
# ---------------------------------------------------------------------------
_US_MEASUREMENT_STAGE = (
    "You are a radiology assistant interpreting a single breast ultrasound image.\n"
    "Delineate the visible lesion as accurately as you can from the image itself and "
    "estimate every quantity below from visual evidence only.\n"
    "\n"
    "UNITS\n"
    "- Treat the displayed image as a pixel grid. Report all lengths in displayed-image "
    "pixels, never mm or cm.\n"
    "- Report angles in degrees (0-90). Report grayscale intensities in pixel values (0-255).\n"
    "\n"
    "LESION SELECTION\n"
    "- One lesion visible: evaluate it. Several visible: evaluate the largest or most "
    "prominent one.\n"
    "- No lesion boundary sufficiently visible: set every measurement and every feature "
    "label to null, and go straight to STAGE 3.\n"
    "\n"
    "STAGE 1 - MEASURE\n"
    "Do this before forming any impression of malignancy. Do not let an expected "
    "diagnosis change a measurement. If an exact value is impossible, give your best "
    "approximate number rather than null.\n"
    "\n"
    "Geometry\n"
    "  long_axis            = longest visible diameter of the lesion\n"
    "  short_axis           = widest diameter perpendicular to long_axis\n"
    "  orientation_degrees  = acute angle between long_axis and the horizontal image "
    "axis (0 = horizontal, 90 = vertical)\n"
    "  bounding_box_width, bounding_box_height = size of the lesion's bounding box\n"
    "  axis_difference_percent = 100 * (long_axis - short_axis) / long_axis\n"
    "\n"
    "Margin conspicuity\n"
    "  Use an inner band just inside the lesion edge and an outer band just outside it, "
    "each about 8% of short_axis wide.\n"
    "  inner_border_median, outer_border_median = typical brightness of each band\n"
    "  boundary_contrast_proxy = |outer_border_median - inner_border_median|\n"
    "  local_noise_estimate    = 1.4826 * median absolute deviation of about 8 sample "
    "points taken around the boundary from both bands, avoiding calipers, text and "
    "artifacts; use at least 1\n"
    "  contrast_to_noise_proxy = boundary_contrast_proxy / local_noise_estimate\n"
    "\n"
    "Border morphology\n"
    "  Imagine the smallest convex hull enclosing the lesion.\n"
    "  solidity_proxy                = lesion area / convex hull area\n"
    "  perimeter_to_hull_ratio_proxy = lesion perimeter / convex hull perimeter\n"
    "\n"
    "Echo\n"
    "  lesion_core         = lesion interior excluding the outermost 10% of short_axis\n"
    "  echo_reference_ring = surrounding tissue out to 20% of short_axis beyond the border\n"
    "  lesion_core_median, echo_reference_median = typical brightness of each\n"
    "  lesion_iqr = approximate interquartile range of brightness inside lesion_core\n"
    "  lesion_to_reference_ratio_proxy     = lesion_core_median / max(echo_reference_median, 1)\n"
    "  lesion_iqr_to_reference_ratio_proxy = lesion_iqr / max(echo_reference_median, 1)\n"
    "\n"
    "Posterior acoustics\n"
    "  posterior_strip = region directly below the lesion, same width as the bounding "
    "box, height 1.5 * bounding_box_height, clipped at the image bottom.\n"
    "  Reference strips = same vertical span, one to the left and one to the right, each "
    "one bounding-box width wide.\n"
    "  posterior_strip_median, posterior_reference_median = typical brightness of each\n"
    "  posterior_to_reference_ratio_proxy = posterior_strip_median / "
    "max(posterior_reference_median, 1)\n"
    "  Divide posterior_strip into 10 equal vertical columns. For each column compare it "
    "with the reference at the same depth and record \"shadowing\" if its ratio <= 0.9, "
    "\"enhancement\" if >= 1.1, otherwise \"neutral\". Report the 10 states in order as "
    "posterior_column_states, then shadowing_fraction, enhancement_fraction, "
    "max_shadowing_run_columns and max_enhancement_run_columns.\n"
    "\n"
    "STAGE 2 - LABEL\n"
    "Assign each label strictly by applying these thresholds to the numbers you just "
    "wrote. Do not override a threshold with intuition. If an input measurement is null, "
    "the label is null.\n"
    "\n"
    "  shape       : \"oval\" if axis_difference_percent > 20, else \"round\"\n"
    "  orientation : \"parallel\" if orientation_degrees < 45, else \"non_parallel\"\n"
    "  margin      : \"circumscribed\" if contrast_to_noise_proxy >= 2.0, "
    "else \"not_circumscribed\"\n"
    "  border      : \"smooth\" if solidity_proxy >= 0.95 and "
    "perimeter_to_hull_ratio_proxy <= 1.05;\n"
    "                \"spiky\" if perimeter_to_hull_ratio_proxy >= 1.12; "
    "else \"lobulated\"\n"
    "  echo_pattern: \"anechoic\" if lesion_to_reference_ratio_proxy <= 0.25 and "
    "lesion_iqr <= 12;\n"
    "                \"hypoechoic\" if lesion_to_reference_ratio_proxy < 0.85;\n"
    "                \"hyperechoic\" if lesion_to_reference_ratio_proxy > 1.15; "
    "else \"isoechoic\"\n"
    "  echotexture : \"heterogeneous\" if lesion_iqr_to_reference_ratio_proxy >= 0.06, "
    "else \"homogeneous\"\n"
    "  posterior_feature:\n"
    "                \"combined_pattern\" if shadowing_fraction >= 0.2 and "
    "enhancement_fraction >= 0.2 and max_shadowing_run_columns >= 3 and "
    "max_enhancement_run_columns >= 3;\n"
    "                \"shadowing\" if posterior_to_reference_ratio_proxy <= 0.9;\n"
    "                \"enhancement\" if posterior_to_reference_ratio_proxy >= 1.1;\n"
    "                else \"no_clear_posterior_feature\"\n"
)

# Points table shared by the three breast decision stages, so the malignancy arm
# and the two BI-RADS arms grade the same evidence identically and differ only
# in how the total is cut. Weights follow BI-RADS malignancy signs, ordered by
# the association each label actually shows on BUSBRA (orientation and border
# carry the signal; margin and echotexture are near-constant, so they are worth
# 1 point, not 3).
_BREAST_SUSPICION_SCORE = (
    "Score every line that matches the labels from STAGE 2 and add them up. A null "
    "label scores 0.\n"
    "    +3  border = spiky\n"
    "    +3  orientation = non_parallel\n"
    "    +2  posterior_feature = shadowing\n"
    "    +1  posterior_feature = combined_pattern\n"
    "    +1  shape = round\n"
    "    +1  margin = not_circumscribed\n"
    "    +1  echotexture = heterogeneous\n"
    "    +1  echo_pattern = hypoechoic\n"
    "    -1  orientation = parallel\n"
    "    -1  posterior_feature = enhancement\n"
    "    -2  border = smooth\n"
    "    -3  echo_pattern = anechoic\n"
    "  Write the total as suspicion_score.\n"
)

_STRUCTURED_JSON_SCHEMA = (
    "{\n"
    '  "geometry": {"long_axis": null, "short_axis": null, '
    '"axis_difference_percent": null, "orientation_degrees": null, '
    '"bounding_box_width": null, "bounding_box_height": null},\n'
    '  "margin_measurements": {"inner_border_median": null, "outer_border_median": null, '
    '"boundary_contrast_proxy": null, "local_noise_estimate": null, '
    '"contrast_to_noise_proxy": null},\n'
    '  "border_measurements": {"solidity_proxy": null, '
    '"perimeter_to_hull_ratio_proxy": null},\n'
    '  "echo_measurements": {"lesion_core_median": null, "echo_reference_median": null, '
    '"lesion_to_reference_ratio_proxy": null, "lesion_iqr": null, '
    '"lesion_iqr_to_reference_ratio_proxy": null},\n'
    '  "posterior_measurements": {"posterior_strip_median": null, '
    '"posterior_reference_median": null, "posterior_to_reference_ratio_proxy": null, '
    '"posterior_column_states": [], "shadowing_fraction": null, '
    '"enhancement_fraction": null, "max_shadowing_run_columns": null, '
    '"max_enhancement_run_columns": null},\n'
    '  "feature_labels": {"shape": null, "orientation": null, "margin": null, '
    '"border": null, "echo_pattern": null, "echotexture": null, '
    '"posterior_feature": null},\n'
    '  "evidence": {"shape": "", "orientation": "", "margin": "", "border": "", '
    '"echo_pattern": "", "echotexture": "", "posterior_feature": ""},\n'
    '  "decision": {"rule_fired": null, "suspicion_score": null, "label": null}\n'
    "}\n"
)


def _structured_output_rules(answer_noun: str) -> str:
    return (
        "OUTPUT\n"
        "Emit a single JSON object in exactly the schema below, then a newline, then the "
        f"final {answer_noun} inside <answer>...</answer>. Nothing else -- no prose "
        "before the JSON, no code fences.\n"
        "Numbers must be numbers, not strings. Round pixel measurements to integers and "
        "ratios to two decimals. Each evidence string is at most 12 words and cites only "
        "what is visible in the image.\n"
        "The value inside <answer> must equal decision.label exactly.\n"
        "\n"
        "Schema:\n"
        f"{_STRUCTURED_JSON_SCHEMA}"
        "<answer>...</answer>\n"
    )


_BREAST_MALIGNANCY_STRUCTURED = (
    _US_MEASUREMENT_STAGE
    + "\n"
    "STAGE 3 - DECIDE\n"
    "Apply these rules in order and stop at the first one that fires. Record which one "
    "fired as rule_fired. Follow the rule even if your overall impression disagrees "
    "with it.\n"
    "\n"
    "  D1  No lesion is visible                                  -> \"normal\"\n"
    "  D2  echo_pattern = anechoic AND margin = circumscribed\n"
    "      AND border = smooth AND posterior_feature is\n"
    "      \"enhancement\" or \"no_clear_posterior_feature\"        -> \"benign\"\n"
    "  D3  border = spiky OR orientation = non_parallel          -> \"malignant\"\n"
    "  D4  Otherwise compute suspicion_score:\n"
    f"{_BREAST_SUSPICION_SCORE}"
    "      suspicion_score >= 4                                  -> \"malignant\"\n"
    "      suspicion_score < 4                                   -> \"benign\"\n"
    "\n"
    "Set suspicion_score to null when D1, D2 or D3 fired.\n"
    "\n"
    "options: benign, malignant, normal\n"
    "\n"
    + _structured_output_rules("option")
)


def _breast_birads_structured(fine: bool) -> str:
    if fine:
        cuts = (
            "      suspicion_score <= 0    -> \"2\"\n"
            "      suspicion_score 1-2     -> \"3\"\n"
            "      suspicion_score 3-4     -> \"4A\"\n"
            "      suspicion_score 5-6     -> \"4B\"\n"
            "      suspicion_score 7-8     -> \"4C\"\n"
            "      suspicion_score >= 9    -> \"5\"\n"
        )
        options = "options: ['2', '3', '4A', '4B', '4C', '5']"
    else:
        cuts = (
            "      suspicion_score <= 0    -> \"2\"\n"
            "      suspicion_score 1-2     -> \"3\"\n"
            "      suspicion_score 3-7     -> \"4\"\n"
            "      suspicion_score >= 8    -> \"5\"\n"
        )
        options = "options: ['2', '3', '4', '5']"
    return (
        _US_MEASUREMENT_STAGE
        + "\n"
        "STAGE 3 - ASSESS\n"
        "Map the STAGE 2 labels to an ACR BI-RADS category using these rules in order. "
        "Stop at the first one that fires and record it as rule_fired. Follow the rule "
        "even if your overall impression disagrees with it.\n"
        "\n"
        "  D1  echo_pattern = anechoic AND margin = circumscribed\n"
        "      AND border = smooth AND posterior_feature is\n"
        "      \"enhancement\" or \"no_clear_posterior_feature\"    -> simple cyst, category \"2\"\n"
        "  D2  Otherwise compute suspicion_score:\n"
        f"{_BREAST_SUSPICION_SCORE}"
        "      then cut it:\n"
        f"{cuts}"
        "\n"
        "Set suspicion_score to null when D1 fired.\n"
        "\n"
        f"{options}\n"
        "\n"
        + _structured_output_rules("category")
    )


PROMPT_TEMPLATES = {
    "generic_medical_classification": {
        "keywords": ["classification", "medical", "image"],
        "modalities": [],
        "anatomies": [],
        "task_name": "medical_image_classification",
        "task_group": "medical_vlm_classification",
        "instruction": "Classify the main finding shown in the image.",
        "label_schema": "Use the dataset label mapping.",
        "direct_format": "Respond only with the final answer inside <answer>...</answer>.",
        "reasoning_format": (
            "Explain the important image findings inside <reasoning>...</reasoning>, "
            "then give the final answer inside <answer>...</answer>."
        ),
    },
    "breast_ultrasound_malignancy": {
        "keywords": [
            "breast",
            "breast ultrasound",
            "ultrasound",
            "us",
            "lesion",
            "malignancy",
            "benign",
            "malignant",
        ],
        "modalities": ["ultrasound"],
        "anatomies": ["breast"],
        "task_name": "disease_diagnosis",
        "task_group": "medical_vlm_classification",
        "instruction": (
            "You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to carefully examine the provided breast ultrasound image, "
            "evaluate any identified lesions or abnormalities based on key sonographic "
            "characteristics (including shape, orientation, margin, echo pattern, "
            "posterior acoustic features, and associated features), synthesize these "
            "features to form an overall impression about the likelihood of malignancy, "
            "and then choose the single best option from the following list that "
            "accurately summarizes this assessment."
        ),
        "label_schema": "options: benign, malignant, normal",
        "direct_prompt": (
            "You are a radiologist analyzing a breast ultrasound image.\n"
            "Your task is to carefully examine the provided breast ultrasound image, "
            "evaluate any identified lesions or abnormalities based on key sonographic "
            "characteristics (including shape, orientation, margin, echo pattern, "
            "posterior acoustic features, and associated features), synthesize these "
            "features to form an overall impression about the likelihood of malignancy, "
            "and then choose the single best option from the following list that "
            "accurately summarizes this assessment.\n\n"
            "options: benign, malignant, normal\n\n"
            "Output format: only the exact text of the chosen option from the list "
            "above. Do not include any introductory phrases, explanations, numbering, "
            "or formatting."
        ),
        "reasoning_prompt": (
            "You are a radiologist analyzing a breast ultrasound image.\n"
            "Your task is to carefully examine the provided breast ultrasound image, "
            "evaluate any identified lesions or abnormalities based on key sonographic "
            "characteristics (including shape, orientation, margin, echo pattern, "
            "posterior acoustic features, and associated features), and reason about "
            "the likelihood of malignancy.\n\n"
            "options: benign, malignant, normal\n\n"
            "Output format:\n"
            "Put the imaging-based reasoning inside <reasoning>...</reasoning>, "
            "using at most 4 short sentences. Do not repeat yourself. "
            "Immediately after closing </reasoning>, continue with <answer> "
            "followed by exactly one of the listed options and then </answer>. "
            "Do not end the response before the <answer> tag has been written, "
            "and do not include any text outside these two tags."
        ),
    },
    "breast_ultrasound_birads": {
        "keywords": [
            "breast",
            "ultrasound",
            "us",
            "birads",
            "bi-rads",
            "acr",
            "assessment",
        ],
        "modalities": ["ultrasound", "us"],
        "anatomies": ["breast"],
        "task_name": "disease_diagnosis",
        "task_group": "medical_vlm_classification",
        "instruction": (
            "You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to synthesize the sonographic characteristics of any "
            "identified lesions (or lack thereof) into a final ACR BI-RADS "
            "(Breast Imaging Reporting and Data System) assessment category."
        ),
        "label_schema": "options: ['2', '3', '4', '4A', '4B', '4C', '5']",
        "direct_prompt": (
            "Prompt: You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to synthesize the sonographic characteristics of any "
            "identified lesions (or lack thereof) into a final ACR BI-RADS "
            "(Breast Imaging Reporting and Data System) assessment category.\n"
            "BI-RADS Ultrasound Assessment Category Definitions\n"
            "- '2' (Benign): Findings are definitively benign (e.g., simple cysts, "
            "intramammary lymph nodes, stable surgical implants/changes). 0% "
            "likelihood of malignancy. Requires routine screening follow-up.\n"
            "- '3' (Probably Benign): Findings have characteristic benign features "
            "but are not definitively benign (e.g., presumed fibroadenoma, "
            "complicated cyst). Very low likelihood of malignancy (<2%). "
            "Short-interval follow-up is typically recommended.\n"
            "- '4A' (Low Suspicion for Malignancy): Findings warrant biopsy but "
            "have a low probability of malignancy (>2% to <=10%).\n"
            "- '4B' (Moderate Suspicion for Malignancy): Findings warrant biopsy "
            "with an intermediate probability of malignancy (>10% to <=50%).\n"
            "- '4C' (High Suspicion for Malignancy): Findings warrant biopsy with "
            "a high probability of malignancy (>50% to <95%), without the classic "
            "features of Category 5.\n"
            "- '5' (Highly Suggestive of Malignancy): Findings have classic "
            "malignant features. Very high probability of malignancy (>=95%).\n"
            "Choose the single most appropriate BI-RADS assessment category from "
            "the options below.\n"
            "options: ['2', '3', '4A', '4B', '4C', '5']\n"
            "Output format: only the exact text of the chosen option from the list "
            "above. Do not include any introductory phrases or explanation."
        ),
        "reasoning_prompt": (
            "Prompt: You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to synthesize the sonographic characteristics of any "
            "identified lesions (or lack thereof) into a final ACR BI-RADS "
            "(Breast Imaging Reporting and Data System) assessment category.\n"
            "Choose the single most appropriate BI-RADS assessment category from "
            "these options: ['2', '3', '4A', '4B', '4C', '5']\n"
            "Output format:\n"
            "Put the imaging-based reasoning inside <reasoning>...</reasoning>, "
            "using at most 4 short sentences. Do not repeat yourself. "
            "Immediately after closing </reasoning>, continue with <answer> "
            "followed by exactly one of the listed BI-RADS categories and then "
            "</answer>. Do not end the response before the <answer> tag has "
            "been written, and do not include any text outside these two tags."
        ),
    },
    "breast_ultrasound_birads_busbra": {
        "keywords": [
            "breast",
            "ultrasound",
            "us",
            "birads",
            "bi-rads",
            "acr",
            "assessment",
            "busbra",
        ],
        "modalities": ["ultrasound", "us"],
        "anatomies": ["breast"],
        "task_name": "disease_diagnosis",
        "task_group": "medical_vlm_classification",
        "instruction": (
            "You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to synthesize the sonographic characteristics of any "
            "identified lesions (or lack thereof) into a final coarse ACR BI-RADS "
            "(Breast Imaging Reporting and Data System) assessment category."
        ),
        "label_schema": "options: ['2', '3', '4', '5']",
        "direct_prompt": (
            "Prompt: You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to synthesize the sonographic characteristics of any "
            "identified lesions (or lack thereof) into a final coarse ACR BI-RADS "
            "(Breast Imaging Reporting and Data System) assessment category.\n"
            "Coarse BI-RADS category definitions\n"
            "- '2' (Benign): Findings are definitively benign.\n"
            "- '3' (Probably Benign): Findings are probably benign and usually merit short-interval follow-up.\n"
            "- '4' (Suspicious Abnormality): Findings are suspicious for malignancy and biopsy should be considered.\n"
            "- '5' (Highly Suggestive of Malignancy): Findings have classic malignant features.\n"
            "Choose the single most appropriate BI-RADS assessment category from "
            "the options below.\n"
            "options: ['2', '3', '4', '5']\n"
            "Output format: only the exact text of the chosen option from the list "
            "above. Do not include any introductory phrases or explanation."
        ),
        "reasoning_prompt": (
            "Prompt: You are a radiologist analyzing a breast ultrasound image. "
            "Your task is to synthesize the sonographic characteristics of any "
            "identified lesions (or lack thereof) into a final coarse ACR BI-RADS "
            "(Breast Imaging Reporting and Data System) assessment category.\n"
            "Choose the single most appropriate BI-RADS assessment category from "
            "these options: ['2', '3', '4', '5']\n"
            "Output format:\n"
            "Put the imaging-based reasoning inside <reasoning>...</reasoning>, "
            "using at most 4 short sentences. Do not repeat yourself. "
            "Immediately after closing </reasoning>, continue with <answer> "
            "followed by exactly one of the listed BI-RADS categories and then "
            "</answer>. Do not end the response before the <answer> tag has "
            "been written, and do not include any text outside these two tags."
        ),
    },
    "thyroid_ultrasound_malignancy": {
        "keywords": [
            "thyroid",
            "thyroid gland",
            "head and neck",
            "ultrasound",
            "us",
            "nodule",
            "malignancy",
            "benign",
            "malignant",
        ],
        "modalities": ["ultrasound", "us"],
        "anatomies": ["thyroid", "thyroid gland"],
        "task_name": "disease_diagnosis",
        "task_group": "medical_vlm_classification",
        "instruction": (
            "You are a radiologist specializing in head and neck or endocrine imaging, "
            "analyzing an ultrasound image of the thyroid gland. "
            "Your task is to carefully examine the provided thyroid ultrasound image, "
            "evaluate the overall thyroid gland parenchyma (echogenicity, texture, vascularity), "
            "identify any focal nodules, assess the specific sonographic features of any nodules "
            "found (including composition, echogenicity, shape, margin, and echogenic foci), "
            "synthesize these findings to determine if the gland appears normal, contains "
            "benign-appearing findings, or contains findings suspicious for malignancy, and then "
            "choose the single best option from the following list that accurately summarizes "
            "this assessment."
        ),
        "label_schema": "options: normal thyroid, benign, malignant",
        "direct_prompt": (
            "You are a radiologist specializing in head and neck or endocrine imaging, analyzing an ultrasound\n"
            "image of the thyroid gland.\n"
            "Your task is to carefully examine the provided thyroid ultrasound image, evaluate the overall thyroid\n"
            "gland parenchyma (echogenicity, texture, vascularity), identify any focal nodules, assess the specific\n"
            "sonographic features of any nodules found (including composition, echogenicity, shape, margin,\n"
            "and echogenic foci), synthesize these findings to determine if the gland appears normal, contains\n"
            "benign-appearing findings, or contains findings suspicious for malignancy, and then choose the single\n"
            "best option from the following list that accurately summarizes this assessment.\n\n"
            "options: normal thyroid, benign, malignant\n\n"
            "Output format: only the exact text of the chosen option from the list above. "
            "Do not include any introductory phrases, explanations, numbering, or formatting."
        ),
        "reasoning_prompt": (
            "You are a radiologist specializing in head and neck or endocrine imaging, analyzing an ultrasound\n"
            "image of the thyroid gland.\n"
            "Your task is to carefully examine the provided thyroid ultrasound image, evaluate the overall thyroid\n"
            "gland parenchyma (echogenicity, texture, vascularity), identify any focal nodules, assess the specific\n"
            "sonographic features of any nodules found (including composition, echogenicity, shape, margin,\n"
            "and echogenic foci), synthesize these findings to determine if the gland appears normal, contains\n"
            "benign-appearing findings, or contains findings suspicious for malignancy.\n\n"
            "options: normal thyroid, benign, malignant\n\n"
            "Output format:\n"
            "Put the imaging-based reasoning inside <reasoning>...</reasoning>, "
            "using at most 4 short sentences. Do not repeat yourself. "
            "Immediately after closing </reasoning>, continue with <answer> "
            "followed by exactly one of the listed options and then </answer>. "
            "Do not end the response before the <answer> tag has been written, "
            "and do not include any text outside these two tags."
        ),
    },
    "liver_ultrasound_malignancy": {
        "keywords": [
            "liver",
            "hepatic",
            "liver ultrasound",
            "ultrasound",
            "us",
            "lesion",
            "malignancy",
            "benign",
            "malignant",
        ],
        "modalities": ["ultrasound", "us"],
        "anatomies": ["liver"],
        "task_name": "disease_diagnosis",
        "task_group": "medical_vlm_classification",
        "instruction": (
            "You are a radiologist analyzing a liver ultrasound image. "
            "Your task is to carefully examine the provided liver ultrasound image, "
            "evaluate the overall liver parenchyma and contour, identify any focal hepatic "
            "lesions or other abnormalities, assess relevant sonographic features of any "
            "detected lesion (including echogenicity, shape, margin, internal architecture, "
            "posterior acoustic features, vascularity if visible, and background liver appearance), "
            "synthesize these features to form an overall impression about the likelihood of "
            "malignancy, and then choose the single best option from the following list that "
            "accurately summarizes this assessment."
        ),
        "label_schema": "options: normal liver, benign, malignant",
        "direct_prompt": (
            "You are a radiologist analyzing a liver ultrasound image.\n"
            "Your task is to carefully examine the provided liver ultrasound image, "
            "evaluate the overall liver parenchyma and contour, identify any focal hepatic "
            "lesions or other abnormalities, assess relevant sonographic features of any "
            "detected lesion (including echogenicity, shape, margin, internal architecture, "
            "posterior acoustic features, vascularity if visible, and background liver appearance), "
            "synthesize these features to form an overall impression about the likelihood of "
            "malignancy, and then choose the single best option from the following list that "
            "accurately summarizes this assessment.\n\n"
            "options: normal liver, benign, malignant\n\n"
            "Output format: only the exact text of the chosen option from the list above. "
            "Do not include any introductory phrases, explanations, numbering, or formatting."
        ),
        "reasoning_prompt": (
            "You are a radiologist analyzing a liver ultrasound image.\n"
            "Your task is to carefully examine the provided liver ultrasound image, "
            "evaluate the overall liver parenchyma and contour, identify any focal hepatic "
            "lesions or other abnormalities, assess relevant sonographic features of any "
            "detected lesion (including echogenicity, shape, margin, internal architecture, "
            "posterior acoustic features, vascularity if visible, and background liver appearance), "
            "and reason about the likelihood of malignancy.\n\n"
            "options: normal liver, benign, malignant\n\n"
            "Output format:\n"
            "Put the imaging-based reasoning inside <reasoning>...</reasoning>, "
            "using at most 4 short sentences. Do not repeat yourself. "
            "Immediately after closing </reasoning>, continue with <answer> "
            "followed by exactly one of the listed options and then </answer>. "
            "Do not end the response before the <answer> tag has been written, "
            "and do not include any text outside these two tags."
        ),
    },
}

DEFAULT_TEMPLATE_KEY = "generic_medical_classification"

# Attached after the fact rather than inlined into each template literal: the
# measurement stage is identical across all three breast tasks and only the
# decision stage differs, so writing it once is what keeps them from drifting.
# Templates without a `structured_prompt` fall back to their direct prompt --
# see build_prompt_variants -- so the structured arm is a no-op, not a crash,
# on the thyroid/liver datasets.
PROMPT_TEMPLATES["breast_ultrasound_malignancy"]["structured_prompt"] = (
    _BREAST_MALIGNANCY_STRUCTURED
)
PROMPT_TEMPLATES["breast_ultrasound_birads"]["structured_prompt"] = (
    _breast_birads_structured(fine=True)
)
PROMPT_TEMPLATES["breast_ultrasound_birads_busbra"]["structured_prompt"] = (
    _breast_birads_structured(fine=False)
)


def normalize_keyword(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def normalize_keywords(values: list[str]) -> set[str]:
    normalized = set()
    for value in values:
        if not value:
            continue
        compact = normalize_keyword(value)
        normalized.add(compact)
        normalized.update(part for part in compact.split() if part)
    return normalized


def keyword_matches(keyword: str, normalized_keywords: set[str]) -> bool:
    normalized_keyword = normalize_keyword(keyword)
    if normalized_keyword in normalized_keywords:
        return True
    parts = [part for part in normalized_keyword.split() if part]
    return bool(parts) and all(part in normalized_keywords for part in parts)


def resolve_prompt_template(
    *,
    keywords: list[str],
    modality: str = "",
    anatomy: str = "",
) -> tuple[str, dict[str, Any]]:
    normalized_keywords = normalize_keywords(keywords + [modality, anatomy])
    normalized_modality = normalize_keyword(modality) if modality else ""
    normalized_anatomy = normalize_keyword(anatomy) if anatomy else ""

    best_key = DEFAULT_TEMPLATE_KEY
    best_score = -1

    for template_key, template in PROMPT_TEMPLATES.items():
        score = 0

        keyword_hits = sum(
            1 for keyword in template.get("keywords", []) if keyword_matches(keyword, normalized_keywords)
        )
        score += keyword_hits * 2

        if normalized_modality and normalized_modality in {
            normalize_keyword(item) for item in template.get("modalities", [])
        }:
            score += 3

        if normalized_anatomy and normalized_anatomy in {
            normalize_keyword(item) for item in template.get("anatomies", [])
        }:
            score += 2

        if score > best_score:
            best_key = template_key
            best_score = score

    return best_key, PROMPT_TEMPLATES[best_key]


def build_prompt(template: dict[str, Any], prompt_style: str) -> str:
    override_key = f"{prompt_style}_prompt"
    if override_key in template:
        return template[override_key]

    lines = [
        f"Task: {template['instruction']}",
        f"Labels: {template['label_schema']}",
    ]
    if prompt_style == "reasoning":
        lines.append(template["reasoning_format"])
    else:
        lines.append(template["direct_format"])
    lines.append("<image>")
    return "\n".join(lines)


def build_prompt_variants(template: dict[str, Any]) -> dict[str, str]:
    direct = build_prompt(template, "direct")
    return {
        "direct_prompt": direct,
        "reasoning_prompt": build_prompt(template, "reasoning"),
        # Only the breast templates define a structured variant. Falling back to
        # the direct prompt keeps a mis-targeted config comparable to zero-shot
        # instead of silently emitting a generic prompt that asks for a JSON
        # schema the decision stage was never written for.
        "structured_prompt": template.get("structured_prompt", direct),
    }
