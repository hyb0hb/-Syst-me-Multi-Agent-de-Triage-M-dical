/**
 * MedTriage AI — Separate Authorized Held-Out Evaluation UI
 */

document.addEventListener("DOMContentLoaded", () => {
    const overviewEl = document.getElementById("overview");
    const evalWorkspaceEl = document.getElementById("evaluationWorkspace");
    const btnModeDemo = document.getElementById("btnModeDemo");
    const btnModeEval = document.getElementById("btnModeEval");
    const navEvaluation = document.getElementById("navEvaluation");
    const demoNavLinks = [
        document.getElementById("navOverview"),
        document.getElementById("navNewAssessment"),
        document.getElementById("navWorkflow"),
        document.getElementById("navPerformance"),
    ].filter(Boolean);

    if (!evalWorkspaceEl) return;

    let evalToken = "";
    let lastPayload = null;
    let lastResult = null;
    let activeErrorTab = "fn";

    const evalAuthForm = document.getElementById("evalAuthForm");
    const evalTokenInput = document.getElementById("evalTokenInput");
    const btnUseDefaultToken = document.getElementById("btnUseDefaultToken");
    const btnLockEval = document.getElementById("btnLockEval");
    const evalAuthError = document.getElementById("evalAuthError");
    const evalAuthStatusPill = document.getElementById("evalAuthStatusPill");
    const evalAuthorizedArea = document.getElementById("evalAuthorizedArea");

    const evalDatasetSelect = document.getElementById("evalDatasetSelect");
    const evalSamplingMode = document.getElementById("evalSamplingMode");
    const evalSampleSize = document.getElementById("evalSampleSize");
    const evalSeed = document.getElementById("evalSeed");
    const evalBootstrap = document.getElementById("evalBootstrap");
    const btnRunEvaluation = document.getElementById("btnRunEvaluation");
    const btnExportMd = document.getElementById("btnExportMd");
    const btnExportJson = document.getElementById("btnExportJson");
    const evalRunningStatus = document.getElementById("evalRunningStatus");
    const evalResultsContainer = document.getElementById("evalResultsContainer");

    const evalErrorSystemSelect = document.getElementById("evalErrorSystemSelect");
    const tabFnBtn = document.getElementById("tabFnBtn");
    const tabFpBtn = document.getElementById("tabFpBtn");

    function esc(v) {
        return String(v ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function switchWorkspace(mode) {
        const isEval = mode === "eval";
        overviewEl.style.display = isEval ? "none" : "flex";
        evalWorkspaceEl.style.display = isEval ? "flex" : "none";
        if (btnModeDemo) btnModeDemo.classList.toggle("active", !isEval);
        if (btnModeEval) btnModeEval.classList.toggle("active", isEval);
        document.querySelectorAll(".sidebar-nav .nav-item").forEach((n) => n.classList.remove("active"));
        if (isEval && navEvaluation) {
            navEvaluation.classList.add("active");
        } else if (!isEval && demoNavLinks[0]) {
            demoNavLinks[0].classList.add("active");
        }
    }

    if (navEvaluation) {
        navEvaluation.addEventListener("click", (e) => {
            e.preventDefault();
            switchWorkspace("eval");
        });
    }
    if (btnModeEval) btnModeEval.addEventListener("click", () => switchWorkspace("eval"));
    if (btnModeDemo) btnModeDemo.addEventListener("click", () => switchWorkspace("demo"));
    demoNavLinks.forEach((el) => {
        el.addEventListener("click", () => switchWorkspace("demo"));
    });

    if (btnUseDefaultToken) {
        btnUseDefaultToken.addEventListener("click", () => {
            evalTokenInput.value = "research-eval-2026";
        });
    }

    if (evalSamplingMode) {
        evalSamplingMode.addEventListener("change", () => {
            evalSampleSize.disabled = evalSamplingMode.value !== "seeded";
        });
    }

    function setAuthorizedState(unlocked) {
        evalAuthorizedArea.style.display = unlocked ? "block" : "none";
        evalAuthStatusPill.textContent = unlocked ? "Authorized Session" : "Locked";
        evalAuthStatusPill.classList.toggle("unlocked", unlocked);
        if (!unlocked) {
            evalToken = "";
            evalResultsContainer.style.display = "none";
        }
    }

    if (btnLockEval) {
        btnLockEval.addEventListener("click", () => {
            evalTokenInput.value = "";
            setAuthorizedState(false);
        });
    }

    evalAuthForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        evalAuthError.style.display = "none";
        const candidate = evalTokenInput.value.trim();
        if (!candidate) return;

        try {
            const res = await fetch("/api/evaluation/metadata", {
                headers: { "X-Eval-Token": candidate },
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            evalToken = candidate;
            setAuthorizedState(true);
        } catch (err) {
            setAuthorizedState(false);
            evalAuthError.textContent = `Authorization failed: ${err.message}`;
            evalAuthError.style.display = "block";
        }
    });

    function buildRequestPayload() {
        const selectedSystems = Array.from(document.querySelectorAll(".eval-sys-check"))
            .filter((cb) => cb.checked)
            .map((cb) => cb.value);
        if (selectedSystems.length === 0) {
            throw new Error("Select at least one system to evaluate.");
        }
        return {
            dataset_id: evalDatasetSelect.value,
            systems: selectedSystems,
            sample_size: evalSamplingMode.value === "seeded" ? parseInt(evalSampleSize.value, 10) : null,
            seed: parseInt(evalSeed.value || "42", 10),
            n_bootstrap: parseInt(evalBootstrap.value || "400", 10),
        };
    }

    btnRunEvaluation.addEventListener("click", async () => {
        if (!evalToken) return;
        let payload;
        try {
            payload = buildRequestPayload();
        } catch (err) {
            alert(err.message);
            return;
        }

        btnRunEvaluation.disabled = true;
        btnExportMd.disabled = true;
        btnExportJson.disabled = true;
        evalRunningStatus.textContent = "Running blind held-out predictions and bootstrap CIs...";

        try {
            const res = await fetch("/api/evaluation/run", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Eval-Token": evalToken,
                },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || `HTTP ${res.status}`);
            }
            lastPayload = payload;
            lastResult = data;
            renderEvaluationResult(data);
            btnExportMd.disabled = false;
            btnExportJson.disabled = false;
            evalRunningStatus.textContent = "Completed cleanly on held-out samples.";
        } catch (err) {
            evalRunningStatus.textContent = `Error: ${err.message}`;
        } finally {
            btnRunEvaluation.disabled = false;
        }
    });

    async function exportReport(format) {
        if (!evalToken || !lastPayload) return;
        const res = await fetch("/api/evaluation/export", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Eval-Token": evalToken,
            },
            body: JSON.stringify({ ...lastPayload, format }),
        });
        if (!res.ok) {
            alert("Export failed.");
            return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `medtriage_eval_${lastPayload.dataset_id}_seed${lastPayload.seed}.${format === "json" ? "json" : "md"}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }

    btnExportMd.addEventListener("click", () => exportReport("markdown"));
    btnExportJson.addEventListener("click", () => exportReport("json"));

    function renderEvaluationResult(res) {
        evalResultsContainer.style.display = "flex";
        evalResultsContainer.style.flexDirection = "column";
        evalResultsContainer.style.gap = "18px";

        const ds = res.dataset;
        const samp = res.sampling;
        const sysList = Object.values(res.systems);
        const versionsText = sysList.map((s) => `${s.short_label}: ${s.model_version}`).join(" | ");

        document.getElementById("evalProvenanceCard").innerHTML = `
            <div class="eval-provenance-grid">
                <div class="eval-prov-item"><strong>Dataset Version</strong><br>${esc(ds.dataset_name)}<br><code>${esc(ds.dataset_version)}</code></div>
                <div class="eval-prov-item"><strong>Model Versions</strong><br><code>${esc(versionsText)}</code></div>
                <div class="eval-prov-item"><strong>Sample Count &amp; IDs Hash</strong><br>${ds.evaluated_count} total (${ds.evaluated_positives} positives)<br><code>seed=${samp.seed} | hash=${esc(ds.sample_ids_hash)}</code></div>
                <div class="eval-prov-item"><strong>Evaluation Date (UTC)</strong><br>${esc(res.evaluation_date_utc)}<br><code>cross_patient_paired=${ds.cross_patient_paired} | simulated=${ds.is_simulated}</code></div>
            </div>
            <p style="margin-top:10px;font-size:0.8rem;color:#475569;">${esc(ds.evaluation_note)}</p>
        `;

        const verdictEl = document.getElementById("evalVerdictBanner");
        const verdictTitles = {
            improvement_demonstrated: "Verdict: Improvement Demonstrated",
            no_improvement_demonstrated: "Verdict: No Improvement Demonstrated",
            insufficient_evidence: "Verdict: Insufficient Evidence",
        };
        verdictEl.className = `eval-verdict-banner ${res.verdict}`;
        verdictEl.innerHTML = `<strong>${esc(verdictTitles[res.verdict] || res.verdict)}</strong><p style="margin-top:4px;font-size:0.86rem;">${esc(res.verdict_reason)}</p>`;

        // Metrics comparison table
        const fmtCi = (m, k) => `${m[k].toFixed(4)}<span class="eval-ci-sub">[${m.ci_95[k][0].toFixed(3)}, ${m.ci_95[k][1].toFixed(3)}]</span>`;
        document.getElementById("evalComparisonTable").innerHTML = `
            <thead>
                <tr>
                    <th>System</th><th>ROC-AUC (95% CI)</th><th>PR-AUC (95% CI)</th>
                    <th>Sensitivity (95% CI)</th><th>Specificity (95% CI)</th>
                    <th>Precision (95% CI)</th><th>F1 (95% CI)</th>
                    <th>Brier / ECE</th><th>Slope / Intercept</th><th>FN / FP</th>
                </tr>
            </thead>
            <tbody>
                ${sysList.map((s) => {
                    const m = s.metrics;
                    return `<tr>
                        <td><strong>${esc(s.short_label)}</strong></td>
                        <td>${fmtCi(m, "roc_auc")}</td>
                        <td>${fmtCi(m, "pr_auc")}</td>
                        <td>${fmtCi(m, "sensitivity")}</td>
                        <td>${fmtCi(m, "specificity")}</td>
                        <td>${fmtCi(m, "precision")}</td>
                        <td>${fmtCi(m, "f1")}</td>
                        <td>${m.brier_score.toFixed(4)} / ${m.ece.toFixed(4)}</td>
                        <td>${m.calibration_slope ?? "—"} / ${m.calibration_intercept ?? "—"}</td>
                        <td><strong>${m.fn}</strong> / ${m.fp}</td>
                    </tr>`;
                }).join("")}
            </tbody>
        `;

        // Pairwise significance
        const pwContainer = document.getElementById("evalPairwiseContainer");
        if (!res.pairwise_comparisons.length) {
            pwContainer.innerHTML = `<p style="font-size:0.84rem;color:#64748b;">Select at least two differing systems on clinical/simulated benchmarks to display paired McNemar &amp; delta bootstrap tests.</p>`;
        } else {
            pwContainer.innerHTML = res.pairwise_comparisons.map((pw) => {
                const dsSens = pw.deltas.sensitivity;
                const dsAuc = pw.deltas.roc_auc;
                const dsF1 = pw.deltas.f1;
                return `<div class="eval-pair-card">
                    <strong>${esc(pw.comparison)}</strong>
                    <p style="margin:6px 0;color:#334155;">${esc(pw.interpretation)}</p>
                    <div>&Delta; Sensitivity: <strong>${dsSens.delta >= 0 ? "+" : ""}${dsSens.delta.toFixed(4)}</strong> [${dsSens.ci_95[0].toFixed(3)}, ${dsSens.ci_95[1].toFixed(3)}]</div>
                    <div>&Delta; ROC-AUC: <strong>${dsAuc.delta >= 0 ? "+" : ""}${dsAuc.delta.toFixed(4)}</strong> [${dsAuc.ci_95[0].toFixed(3)}, ${dsAuc.ci_95[1].toFixed(3)}]</div>
                    <div>&Delta; F1: <strong>${dsF1.delta >= 0 ? "+" : ""}${dsF1.delta.toFixed(4)}</strong> [${dsF1.ci_95[0].toFixed(3)}, ${dsF1.ci_95[1].toFixed(3)}]</div>
                    <div style="margin-top:4px;color:#64748b;">McNemar overall p=${pw.mcnemar_overall_p} | sens p=${pw.mcnemar_sensitivity_p}</div>
                </div>`;
            }).join("");
        }

        // Confusion matrices + calibration summary
        document.getElementById("evalMatricesGrid").innerHTML = sysList.map((s) => {
            const m = s.metrics;
            return `<div class="eval-cm-card">
                <strong>${esc(s.short_label)}</strong> (threshold=${m.threshold})
                <table class="eval-cm-table">
                    <tr><th></th><th>Pred Negative</th><th>Pred Urgent</th></tr>
                    <tr><th>True Negative</th><td class="eval-cm-tn">TN: ${m.tn}</td><td class="eval-cm-fp">FP: ${m.fp}</td></tr>
                    <tr><th>True Positive</th><td class="eval-cm-fn">FN: ${m.fn}</td><td class="eval-cm-tp">TP: ${m.tp}</td></tr>
                </table>
                <p style="font-size:0.75rem;color:#475569;margin-top:8px;">
                    Brier=${m.brier_score.toFixed(4)} | ECE=${m.ece.toFixed(4)} | Calibration bins (${m.calibration_bins.length})
                </p>
            </div>`;
        }).join("");

        // Subgroups (render from last selected system, typically System C, or first)
        const subCard = document.getElementById("evalSubgroupsCard");
        const primarySys = res.systems.system_c || sysList[sysList.length - 1];
        if (primarySys && primarySys.subgroups && primarySys.subgroups.length) {
            subCard.style.display = "block";
            document.getElementById("evalSubgroupsTable").innerHTML = `
                <thead><tr><th>Subgroup (${esc(primarySys.short_label)})</th><th>n (Positives)</th><th>Status</th><th>ROC-AUC</th><th>Sensitivity</th><th>Specificity</th><th>F1</th></tr></thead>
                <tbody>
                    ${primarySys.subgroups.map((sg) => sg.supported ? `<tr>
                        <td><strong>${esc(sg.label)}</strong></td><td>${sg.n_samples} (${sg.n_positive})</td>
                        <td>Supported</td><td>${sg.roc_auc.toFixed(4)}</td><td>${sg.sensitivity.toFixed(4)}</td>
                        <td>${sg.specificity.toFixed(4)}</td><td>${sg.f1.toFixed(4)}</td>
                    </tr>` : `<tr style="background:#fffbeb;color:#92400e;">
                        <td><strong>${esc(sg.label)}</strong></td><td>${sg.n_samples} (${sg.n_positive})</td>
                        <td colspan="5">${esc(sg.reason)}</td>
                    </tr>`).join("")}
                </tbody>
            `;
        } else {
            subCard.style.display = "none";
        }

        // Populate system selector for FP / FN review
        evalErrorSystemSelect.innerHTML = sysList.map((s) => `<option value="${esc(s.system_id)}">${esc(s.short_label)} (FN=${s.false_negatives_count}, FP=${s.false_positives_count})</option>`).join("");
        renderErrorTable();

        // Limitations
        document.getElementById("evalLimitationsList").innerHTML = res.limitations.map((l) => `<li>${esc(l)}</li>`).join("");
    }

    function renderErrorTable() {
        if (!lastResult) return;
        const sysId = evalErrorSystemSelect.value;
        const sdata = lastResult.systems[sysId];
        if (!sdata) return;
        const rows = activeErrorTab === "fn" ? sdata.false_negatives : sdata.false_positives;
        document.getElementById("evalErrorsTable").innerHTML = `
            <thead>
                <tr><th>Sample ID</th><th>Ground Truth</th><th>Predicted Proba</th><th>Agent 1 / Agent 2</th><th>Triage Tier</th><th>Non-Identifying Summary</th></tr>
            </thead>
            <tbody>
                ${rows.length ? rows.map((r) => `<tr>
                    <td><code>${esc(r.sample_id)}</code></td>
                    <td>${r.true_label === 1 ? "Positive (1)" : "Negative (0)"}</td>
                    <td><strong>${(r.predicted_proba * 100).toFixed(1)}%</strong></td>
                    <td>${(r.agent1_proba * 100).toFixed(1)}% / ${(r.agent2_proba * 100).toFixed(1)}%</td>
                    <td>${esc(r.triage_level)}</td>
                    <td>${esc(r.summary)}</td>
                </tr>`).join("") : `<tr><td colspan="6">No cases in this error category.</td></tr>`}
            </tbody>
        `;
    }

    evalErrorSystemSelect.addEventListener("change", renderErrorTable);
    tabFnBtn.addEventListener("click", () => {
        activeErrorTab = "fn";
        tabFnBtn.classList.add("active");
        tabFpBtn.classList.remove("active");
        renderErrorTable();
    });
    tabFpBtn.addEventListener("click", () => {
        activeErrorTab = "fp";
        tabFpBtn.classList.add("active");
        tabFnBtn.classList.remove("active");
        renderErrorTable();
    });
});
