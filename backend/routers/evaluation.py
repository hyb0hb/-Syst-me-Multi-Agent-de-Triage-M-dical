"""Router FastAPI protégé pour l'évaluation scientifique reproductible MedTriage AI."""

import hmac
import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from backend.schemas.evaluation import EvaluationRequest, ExportEvaluationRequest
from backend.services.evaluation_service import (
    format_evaluation_markdown,
    get_evaluation_metadata,
    run_reproducible_evaluation,
)

router = APIRouter(prefix="/api/evaluation", tags=["Évaluation Reproductible"])


def get_expected_eval_token() -> str:
    return os.environ.get("MEDTRIAGE_EVAL_TOKEN", "research-eval-2026")


def require_authorized_evaluator(
    x_eval_token: Optional[str] = Header(None, alias="X-Eval-Token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """Restreint les endpoints d'évaluation aux utilisateurs autorisés."""
    provided = x_eval_token
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    expected = get_expected_eval_token()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès évaluation refusé : jeton d'évaluateur autorisé requis (en-tête X-Eval-Token).",
        )
    return "authorized_researcher"


@router.get(
    "/metadata",
    summary="Métadonnées des jeux tenus à l'écart et versions systèmes",
)
def evaluation_metadata(_user: str = Depends(require_authorized_evaluator)):
    return get_evaluation_metadata()


@router.post(
    "/run",
    summary="Exécuter une évaluation reproductible sur données tenues à l'écart",
)
def evaluation_run(
    req: EvaluationRequest,
    _user: str = Depends(require_authorized_evaluator),
):
    try:
        return run_reproducible_evaluation(
            dataset_id=req.dataset_id,
            systems=list(req.systems),
            sample_size=req.sample_size,
            seed=req.seed,
            n_bootstrap=req.n_bootstrap,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(err))
    except FileNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))


@router.post(
    "/export",
    summary="Exporter un rapport d'évaluation reproductible (Markdown ou JSON)",
)
def evaluation_export(
    req: ExportEvaluationRequest,
    _user: str = Depends(require_authorized_evaluator),
):
    result = run_reproducible_evaluation(
        dataset_id=req.dataset_id,
        systems=list(req.systems),
        sample_size=req.sample_size,
        seed=req.seed,
        n_bootstrap=req.n_bootstrap,
    )
    stamp = result["evaluation_date_utc"][:10]
    if req.format == "json":
        filename = f"medtriage_eval_{req.dataset_id}_seed{req.seed}_{stamp}.json"
        return Response(
            content=json.dumps(result, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    filename = f"medtriage_eval_{req.dataset_id}_seed{req.seed}_{stamp}.md"
    return Response(
        content=format_evaluation_markdown(result),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
