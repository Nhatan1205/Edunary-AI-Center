"""
Quality check service — multi-agent pipeline.

Agents:
  1. Curriculum Builder : Format curriculum structure tree text.
  2. Metadata Analyzer  : LLM checks course metadata against quality rules.
  3. Content Analyzer   : LLM checks content batches (lectures, quizzes, assignments) in parallel.
  4. Validator          : Senior auditor LLM pass to filter hallucinated issues.
  5. Formatter          : Deterministic parser to return structured Pydantic models.
"""

from __future__ import annotations

import asyncio
import json

from app.core.logging import get_logger, timer
from app.llm.factory import create_llm, create_llm_from_config
from app.prompts.quality_check_prompts import (
    METADATA_ANALYZER_SYSTEM_PROMPT,
    METADATA_ANALYZER_USER_PROMPT,
    CONTENT_ANALYZER_SYSTEM_PROMPT,
    CONTENT_ANALYZER_USER_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_PROMPT,
)
from app.schemas.quality_check_schema import (
    QualityCheckRequest,
    QualityCheckResponse,
    AiCheckIssue,
)

logger = get_logger(__name__)


async def run_quality_check(request: QualityCheckRequest) -> QualityCheckResponse:
    """
    Orchestrate the quality check pipeline (Metadata Analyzer + Parallel Content Analyzers + Validator).

    Returns:
        QualityCheckResponse with validated issues and analysis summary.
    """
    from langchain_community.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        with timer("Quality check pipeline"):

            # 1. Curriculum Builder
            logger.info("[QualityCheck] Agent 1: Curriculum Builder")
            curriculum_text = _build_curriculum_text(request)
            logger.info(f"[QualityCheck] Curriculum text ready: {len(curriculum_text)} chars")

            # 2. Metadata Analyzer
            logger.info("[QualityCheck] Agent 2: Metadata Analyzer LLM")
            raw_metadata_json = await _run_metadata_analyzer(request, curriculum_text)
            
            try:
                metadata_data = json.loads(raw_metadata_json)
                proposed_issues = metadata_data.get("issues", [])
            except Exception as e:
                logger.warning(f"[QualityCheck] Failed to parse metadata issues: {e}")
                proposed_issues = []

            # 3. Content Analyzer 
            logger.info(f"[QualityCheck] Agent 3: Content Analyzer — processing {len(request.content_batches)} batches")
            if request.content_batches:
                llm = _get_llm(request.llm_config)
                
                # Create parallel tasks
                tasks = [
                    _run_content_analyzer(request, batch, llm)
                    for batch in request.content_batches
                ]
                
                # Execute parallel LLM calls concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Merge results
                for res in batch_results:
                    if isinstance(res, list):
                        proposed_issues.extend(res)
                    else:
                        logger.error(f"[QualityCheck] Content analyzer batch task failed with error: {res}")

            # 4: Validator Pass 
            logger.info("[QualityCheck] Agent 4: Validator LLM (Anti-Hallucination pass)")
            course_content_text = ""
            for batch in request.content_batches:
                course_content_text += f"\n--- BATCH SECTIONS: {', '.join(batch.sections)} ---\n{batch.content_text}\n"

            validated_issues, summary = await _run_validator(
                request=request,
                proposed_issues=proposed_issues,
                curriculum_text=curriculum_text,
                course_content_text=course_content_text,
            )

            # 5: Formatter
            logger.info("[QualityCheck] Agent 5: Formatter")
            validated_json = json.dumps({"issues": validated_issues, "analysis_summary": summary}, ensure_ascii=False)
            issues, final_summary = _parse_and_format(validated_json)

        logger.info("==================================================================")
        logger.info("                  [FLOW TOKEN CONSUMPTION SUMMARY]                ")
        logger.info(f" * Prompt Tokens:     {cb.prompt_tokens}")
        logger.info(f" * Completion Tokens: {cb.completion_tokens}")
        logger.info(f" * Total Tokens:      {cb.total_tokens}")
        logger.info(f" * Estimated Cost:    ${cb.total_cost:.6f} USD")
        logger.info("==================================================================")

    logger.info(f"[QualityCheck] Pipeline complete: {len(issues)} validated issues found")
    return QualityCheckResponse(issues=issues, analysis_summary=final_summary)


async def run_quality_check_diff(request: QualityCheckRequest) -> QualityCheckResponse:
    """
    Orchestrate the diff-based quality check pipeline.
    Bypasses Metadata Analyzer if no metadata-related fields are changed.
    Analyzes only changed content batches in parallel.
    """
    from langchain_community.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        with timer("Quality check diff pipeline"):

            # 1. Curriculum Builder
            logger.info("[QualityCheckDiff] Agent 1: Curriculum Builder")
            curriculum_text = _build_curriculum_text(request)

            # Check if any metadata-related fields were modified
            metadata_fields = {
                "title", "subtitle", "description", "category", "topics", 
                "learning_objectives", "requirements", "target_audience", "curriculum",
                "level", "course_image"
            }
            
            has_metadata_changes = False
            if request.changed_fields:
                normalized_changed = {f.lower().replace(" ", "_") for f in request.changed_fields}
                has_metadata_changes = any(f in metadata_fields for f in normalized_changed)

            proposed_issues = []

            # 2. Run Metadata Analyzer if there are metadata-related changes
            if has_metadata_changes:
                logger.info("[QualityCheckDiff] Agent 2: Metadata Analyzer LLM")
                raw_metadata_json = await _run_metadata_analyzer(request, curriculum_text)
                try:
                    metadata_data = json.loads(raw_metadata_json)
                    proposed_issues = metadata_data.get("issues", [])
                except Exception as e:
                    logger.warning(f"[QualityCheckDiff] Failed to parse metadata issues: {e}")
            else:
                logger.info("[QualityCheckDiff] Skipping Metadata Analyzer (No metadata changes in diff mode)")

            # 3. Content Analyzer
            logger.info(f"[QualityCheckDiff] Agent 3: Content Analyzer — processing {len(request.content_batches)} changed batches")
            if request.content_batches:
                llm = _get_llm(request.llm_config)
                
                # Create parallel tasks
                tasks = [
                    _run_content_analyzer(request, batch, llm)
                    for batch in request.content_batches
                ]
                
                # Execute parallel LLM calls concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Merge results
                for res in batch_results:
                    if isinstance(res, list):
                        proposed_issues.extend(res)
                    else:
                        logger.error(f"[QualityCheckDiff] Content analyzer batch task failed with error: {res}")

            # 4. Validator Pass
            validated_issues = []
            summary = "Diff quality check complete. No issues found in the changed content."

            if proposed_issues:
                logger.info("[QualityCheckDiff] Agent 4: Validator LLM (Anti-Hallucination pass)")
                course_content_text = ""
                for batch in request.content_batches:
                    course_content_text += f"\n--- BATCH SECTIONS: {', '.join(batch.sections)} ---\n{batch.content_text}\n"

                validated_issues, summary = await _run_validator(
                    request=request,
                    proposed_issues=proposed_issues,
                    curriculum_text=curriculum_text,
                    course_content_text=course_content_text,
                )
            else:
                logger.info("[QualityCheckDiff] Skipping Validator (No proposed issues found)")

            # 5. Formatter
            logger.info("[QualityCheckDiff] Agent 5: Formatter")
            validated_json = json.dumps({"issues": validated_issues, "analysis_summary": summary}, ensure_ascii=False)
            issues, final_summary = _parse_and_format(validated_json)


        logger.info("==================================================================")
        logger.info("               [FLOW DIFF TOKEN CONSUMPTION SUMMARY]              ")
        logger.info(f" * Prompt Tokens:     {cb.prompt_tokens}")
        logger.info(f" * Completion Tokens: {cb.completion_tokens}")
        logger.info(f" * Total Tokens:      {cb.total_tokens}")
        logger.info(f" * Estimated Cost:    ${cb.total_cost:.6f} USD")
        logger.info("==================================================================")

    logger.info(f"[QualityCheckDiff] Diff pipeline complete: {len(issues)} validated issues found")
    return QualityCheckResponse(issues=issues, analysis_summary=final_summary)


# ─── Agent 1: Curriculum Builder ──────────────────────────────────────────────

def _build_curriculum_text(request: QualityCheckRequest) -> str:
    """
    Format the curriculum sections and items into a human-readable text block.
    """
    lines: list[str] = []
    for idx, section in enumerate(request.curriculum, 1):
        lines.append(f"Section {idx}: {section.section_title}")
        if section.learning_objectives:
            lines.append(f"  Learning Objectives: {section.learning_objectives}")
        for item_idx, item in enumerate(section.items, 1):
            lines.append(f"    Lecture {item_idx}: {item.title} ({item.content_type})")
    return "\n".join(lines)


# ─── Agent 2: Metadata Analyzer LLM ──────────────────────────────────────────

async def _run_metadata_analyzer(request: QualityCheckRequest, curriculum_text: str) -> str:
    """
    Metadata LLM pass: check course metadata + curriculum against quality rules.
    """
    lo_text  = "\n".join(f"- {lo}" for lo in request.learning_objectives) or "None"
    req_text = "\n".join(f"- {r}"  for r in request.requirements)         or "None"
    aud_text = "\n".join(f"- {a}"  for a in request.target_audience)      or "None"
    topics_text = ", ".join(request.course_topics) or "None"

    user_prompt = METADATA_ANALYZER_USER_PROMPT.format(
        course_title=request.course_title,
        course_subtitle=request.course_subtitle,
        course_description=request.course_description,
        course_category=request.course_category,
        course_topics=topics_text,
        learning_objectives=lo_text,
        requirements=req_text,
        target_audience=aud_text,
        curriculum_text=curriculum_text,
    )

    llm = _get_llm(request.llm_config)

    system_prompt = METADATA_ANALYZER_SYSTEM_PROMPT
    if request.check_mode == "changes_only":
        system_prompt += f"\n\n[CRITICAL DIRECTIVE - DIFF REVIEW MODE]\n" \
                         f"You are reviewing in DIFF MODE (changes only).\n" \
                         f"You MUST only analyze and report issues for the following fields that were actually modified:\n" \
                         f"{', '.join(request.changed_fields)}\n" \
                         f"Do NOT report any issues for unmodified fields, even if they violate other rules. Strictly ignore unmodified sections/fields."

    messages = [
        ("system", system_prompt),
        ("human",  user_prompt),
    ]
    response = await llm.ainvoke(messages)
    raw = _strip_fences(response.content.strip())
    logger.info(f"[Metadata Analyzer] Response: {len(raw)} chars")
    
    # Print tokens if available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.info(f"[Metadata Analyzer] Token usage: {usage}")

    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[Metadata Analyzer] Invalid JSON: {raw[:400]}")
        raise ValueError(f"Metadata Analyzer LLM returned invalid JSON: {e}") from e

    return raw


# ─── Agent 3: Content Analyzer (Parallel Batches) ────────────────────────────

async def _run_content_analyzer(
    request: QualityCheckRequest,
    batch,
    llm,
) -> list[dict]:
    """
    Check quality and policy compliance on a single content batch.
    """
    lo_text = "\n".join(f"- {lo}" for lo in request.learning_objectives) or "None"
    
    user_prompt = CONTENT_ANALYZER_USER_PROMPT.format(
        course_title=request.course_title,
        learning_objectives=lo_text,
        sections_list=", ".join(batch.sections),
        batch_content_text=batch.content_text,
    )
    
    system_prompt = CONTENT_ANALYZER_SYSTEM_PROMPT
    if request.check_mode == "changes_only":
        system_prompt += f"\n\n[CRITICAL DIRECTIVE - DIFF REVIEW MODE]\n" \
                         f"You are reviewing in DIFF MODE (changes only).\n" \
                         f"You MUST only report issues found in the newly added or modified content in this batch.\n" \
                         f"Do NOT report any issues for unmodified text or existing content. Focus exclusively on the new/changed parts."

    messages = [
        ("system", system_prompt),
        ("human",  user_prompt),
    ]
    
    response = await llm.ainvoke(messages)
    raw = _strip_fences(response.content.strip())
    logger.info(f"[Content Analyzer Batch {batch.batch_index}] Response: {len(raw)} chars")
    
    # Print tokens if available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.info(f"[Content Analyzer Batch {batch.batch_index}] Token usage: {usage}")
    
    try:
        data = json.loads(raw)
        return data.get("issues", [])
    except Exception as e:
        logger.warning(f"[Content Analyzer Batch {batch.batch_index}] Failed to parse JSON: {e}")
        return []


# ─── Agent 4: Validator Pass ──────────────────────────────────────────────────

async def _run_validator(
    request: QualityCheckRequest,
    proposed_issues: list[dict],
    curriculum_text: str,
    course_content_text: str,
) -> tuple[list[dict], str]:
    """
    Auditor LLM validator pass: checks and filters proposed issues against metadata
    and full course content to eliminate hallucinations.
    """
    if not proposed_issues:
        return [], "Analysis completed. No issues found."

    llm = _get_llm(request.llm_config)
    
    lo_text  = "\n".join(f"- {lo}" for lo in request.learning_objectives) or "None"
    req_text = "\n".join(f"- {r}"  for r in request.requirements)         or "None"
    aud_text = "\n".join(f"- {a}"  for a in request.target_audience)      or "None"
    topics_text = ", ".join(request.course_topics) or "None"
    
    proposed_json = json.dumps({"issues": proposed_issues}, ensure_ascii=False, indent=2)
    
    user_prompt = VALIDATOR_USER_PROMPT.format(
        course_title=request.course_title,
        course_subtitle=request.course_subtitle,
        course_description=request.course_description,
        course_category=request.course_category,
        course_topics=topics_text,
        learning_objectives=lo_text,
        requirements=req_text,
        target_audience=aud_text,
        curriculum_text=curriculum_text,
        course_content_text=course_content_text,
        proposed_issues_json=proposed_json,
    )
    
    system_prompt = VALIDATOR_SYSTEM_PROMPT
    if request.check_mode == "changes_only":
        system_prompt += f"\n\n[CRITICAL DIRECTIVE - DIFF REVIEW MODE]\n" \
                         f"You are reviewing in DIFF MODE (changes only).\n" \
                         f"You MUST filter out any issues that do not relate to the changed fields: {', '.join(request.changed_fields)} or the changed curriculum content in this batch.\n" \
                         f"Keep only issues that are directly caused by the new changes. Remove all other issues."

    messages = [
        ("system", system_prompt),
        ("human",  user_prompt),
    ]
    
    response = await llm.ainvoke(messages)
    raw = _strip_fences(response.content.strip())
    logger.info(f"[Validator] Response: {len(raw)} chars")
    
    # Print tokens if available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.info(f"[Validator] Token usage: {usage}")
    
    try:
        data = json.loads(raw)
        return data.get("issues", []), data.get("analysis_summary", "Analysis completed.")
    except Exception as e:
        logger.warning(f"[Validator] Failed to parse Validator JSON: {e} — falling back to proposed issues")
        return proposed_issues, "Analysis completed (validator fallback)."


# ─── Agent 5: Formatter ───────────────────────────────────────────────────────

def _parse_and_format(raw_json: str) -> tuple[list[AiCheckIssue], str]:
    """
    Parse validated JSON → Pydantic models.
    Deterministic — no LLM involved.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Formatter could not parse analyzer JSON: {e}") from e

    issues: list[AiCheckIssue] = []
    for item in data.get("issues", []):
        if not item.get("description"):
            logger.warning(f"[Formatter] Skipping issue with no description: {item}")
            continue

        severity = item.get("severity", "suggestion").lower()
        if severity not in {"critical", "warning", "suggestion"}:
            logger.warning(f"[Formatter] Unknown severity '{severity}', defaulting to 'suggestion'")
            severity = "suggestion"

        issues.append(AiCheckIssue(
            ruleId=item.get("ruleId") or item.get("rule_id") or "AI-GEN",
            category=item.get("category", "Other"),
            severity=severity,
            location=item.get("location", ""),
            description=item.get("description", ""),
            evidence=item.get("evidence", ""),
            suggestion=item.get("suggestion", ""),
        ))

    summary = data.get("analysis_summary", "Analysis completed.")
    return issues, summary


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_llm(llm_config):
    """Select LLM — backend-provided config, fall back to .env default."""
    if llm_config and llm_config.api_base:
        logger.info(f"[LLM] Using backend config: model={llm_config.model_name}")
        return create_llm_from_config(llm_config.model_dump())
    logger.info("[LLM] Using default .env LLM provider")
    return create_llm()


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences if LLM wraps output in ```json ... ```."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return raw
