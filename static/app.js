/**
 * MedTriage AI — Interactive Dashboard Application Script
 * Compliant with UI/UX Design Specification & Visual Reference
 */

document.addEventListener("DOMContentLoaded", () => {
    // State variables
    let uploadedFile = null;
    let currentSampleName = "pneumonia.jpg";
    let presetsData = {};
    let sessionAssessments = [];
    let currentWorkflowStep = 1;

    // Elements : Status & Navigation
    const apiStatusDot = document.getElementById("apiStatusDot");
    const apiStatusText = document.getElementById("apiStatusText");

    // Elements : Workflow
    const workflowStepText = document.getElementById("workflowStepText");
    const stepDots = document.querySelectorAll(".step-dot");
    const stageItems = document.querySelectorAll(".workflow-stage-item");

    // Elements : Form & Upload
    const assessmentForm = document.getElementById("assessmentForm");
    const dropzone = document.getElementById("dropzone");
    const imageFileInput = document.getElementById("imageFileInput");
    const xrayPreviewImg = document.getElementById("xrayPreviewImg");
    const previewEmptyState = document.getElementById("previewEmptyState");
    const btnClearPreview = document.getElementById("btnClearPreview");
    const sampleImageNameInput = document.getElementById("sampleImageName");
    const btnStartAnalysis = document.getElementById("btnStartAnalysis");
    const btnSpinner = document.getElementById("btnSpinner");
    const presetChips = document.querySelectorAll(".preset-chip-btn");
    const btnRandomDataset = document.getElementById("btnRandomDataset");
    const datasetSourceBanner = document.getElementById("datasetSourceBanner");
    const datasetSourceText = document.getElementById("datasetSourceText");

    // Elements : Form Inputs
    const ageInput = document.getElementById("ageInput");
    const bmiInput = document.getElementById("bmiInput");
    const genderSelect = document.getElementById("genderSelect");
    const smokingSelect = document.getElementById("smokingSelect");
    const hypertensionSelect = document.getElementById("hypertensionSelect");
    const heartDiseaseSelect = document.getElementById("heartDiseaseSelect");
    const glucoseInput = document.getElementById("glucoseInput");
    const everMarriedInput = document.getElementById("everMarriedInput");
    const workTypeInput = document.getElementById("workTypeInput");
    const residenceInput = document.getElementById("residenceInput");

    // Elements : Results
    const resultsEmptyState = document.getElementById("resultsEmptyState");
    const resultsPopulatedState = document.getElementById("resultsPopulatedState");
    const triageBanner = document.getElementById("triageBanner");
    const triageLevelBadge = document.getElementById("triageLevelBadge");
    const triageTimeframe = document.getElementById("triageTimeframe");
    const triageScoreNumber = document.getElementById("triageScoreNumber");
    const triageConduite = document.getElementById("triageConduite");

    const agent1ScoreText = document.getElementById("agent1ScoreText");
    const agent1Bar = document.getElementById("agent1Bar");
    const agent1StatusLabel = document.getElementById("agent1StatusLabel");

    const agent2ScoreText = document.getElementById("agent2ScoreText");
    const agent2Bar = document.getElementById("agent2Bar");
    const agent2StatusLabel = document.getElementById("agent2StatusLabel");

    const agent3ScoreText = document.getElementById("agent3ScoreText");
    const formulaChipBox = document.getElementById("formulaChipBox");

    // Elements : Recent Assessments
    const recentEmptyState = document.getElementById("recentEmptyState");
    const recentListContainer = document.getElementById("recentListContainer");

    // Elements : Chart & Metrics
    const perfTabs = document.querySelectorAll(".perf-tab-btn");
    const chartBarCnn = document.getElementById("chartBarCnn");
    const chartBarRf = document.getElementById("chartBarRf");
    const chartBarLr = document.getElementById("chartBarLr");
    const chartValCnn = document.getElementById("chartValCnn");
    const chartValRf = document.getElementById("chartValRf");
    const chartValLr = document.getElementById("chartValLr");

    // Données de métriques 100% dynamiques issues du résultat serveur (aucune valeur en dur)
    let metricsData = {};
    let activeMetricTab = "roc_auc";

    // Helper: Extraire de manière résiliente une métrique depuis un objet brut
    function getMetricValue(agentData, metricKey) {
        if (!agentData) return null;
        if (metricKey === "roc_auc" && agentData.roc_auc != null) {
            return parseFloat(agentData.roc_auc);
        }
        const rep = agentData.report || {};
        if (metricKey === "accuracy") {
            if (rep.accuracy != null) return parseFloat(rep.accuracy);
            if (rep["weighted avg"]?.recall != null) return parseFloat(rep["weighted avg"].recall);
        }
        // Trouver la classe d'intérêt (Urgent, PNEUMONIA, 1)
        const targetClass = rep["Urgent"] || rep["PNEUMONIA"] || rep["1"] || rep["urgent"] || rep["pneumonia"];
        if (targetClass) {
            if (metricKey === "recall" && targetClass.recall != null) return parseFloat(targetClass.recall);
            if (metricKey === "precision" && targetClass.precision != null) return parseFloat(targetClass.precision);
            if (metricKey === "f1" && (targetClass["f1-score"] != null || targetClass.f1 != null)) {
                return parseFloat(targetClass["f1-score"] ?? targetClass.f1);
            }
        }
        // Fallback macro avg
        if (rep["macro avg"]) {
            if (metricKey === "recall" && rep["macro avg"].recall != null) return parseFloat(rep["macro avg"].recall);
            if (metricKey === "precision" && rep["macro avg"].precision != null) return parseFloat(rep["macro avg"].precision);
            if (metricKey === "f1" && (rep["macro avg"]["f1-score"] != null || rep["macro avg"].f1 != null)) {
                return parseFloat(rep["macro avg"]["f1-score"] ?? rep["macro avg"].f1);
            }
        }
        return null;
    }

    // ========================================================
    // 1. API Health Check & Dynamic Metrics Synchronization
    // ========================================================
    async function checkApiConnection() {
        try {
            const res = await fetch("/api/metrics");
            if (res.ok) {
                const data = await res.json();
                apiStatusDot.className = "status-dot green";
                apiStatusText.textContent = "FastAPI · Connected";

                // 1. Alimentation 100% dynamique depuis le résultat calculé par le serveur
                if (data.summary) {
                    metricsData = data.summary;
                } else {
                    metricsData = {};
                    const metricKeys = ["roc_auc", "accuracy", "recall", "precision", "f1"];
                    for (const m of metricKeys) {
                        const cnnV = getMetricValue(data.image, m);
                        const rfV = getMetricValue(data.clinique, m);
                        const lrV = getMetricValue(data.decision, m);
                        if (cnnV !== null || rfV !== null || lrV !== null) {
                            metricsData[m] = { cnn: cnnV, rf: rfV, lr: lrV };
                        }
                    }
                }

                // 2. Synchronisation de la formule de consensus de l'Agent 3 (sans modifier le code)
                if (formulaChipBox) {
                    if (data.decision_formula) {
                        formulaChipBox.innerHTML = `<code>${data.decision_formula}</code>`;
                    } else if (data.decision?.coefficients) {
                        const c = data.decision.coefficients;
                        const cImg = parseFloat(c.image ?? 2.99).toFixed(2);
                        const cClin = parseFloat(c.clinique ?? 4.24).toFixed(2);
                        const cInt = parseFloat(c.intercept ?? -3.31);
                        const sign = cInt >= 0 ? "+" : "-";
                        formulaChipBox.innerHTML = `<code>z = (${cImg} × P_img) + (${cClin} × P_clin) ${sign} ${Math.abs(cInt).toFixed(2)}</code>`;
                    }
                }

                // 3. Mise à jour dynamique du graphique et des cartes de référence
                updateChart(activeMetricTab);
            } else {
                throw new Error("HTTP error " + res.status);
            }
        } catch (err) {
            apiStatusDot.className = "status-dot";
            apiStatusDot.style.background = "#E76C72";
            apiStatusText.textContent = "FastAPI · Offline";
        }
    }

    // ========================================================
    // 2. Interactive Workflow Stages (Parallel Multi-Agent)
    // ========================================================
    function setWorkflowStep(stepNumber) {
        currentWorkflowStep = stepNumber;
        if (workflowStepText) {
            workflowStepText.textContent = `Step ${stepNumber} of 4`;
        }

        stepDots.forEach((dot, idx) => {
            dot.classList.remove("active", "completed");
            if (idx + 1 === stepNumber) {
                dot.classList.add("active");
            } else if (idx + 1 < stepNumber) {
                dot.classList.add("completed");
            }
        });

        const forkConnector = document.querySelector(".fork-connector");
        const joinConnector = document.querySelector(".join-connector");

        stageItems.forEach((item) => {
            const s = parseInt(item.dataset.stage, 10);
            if (stepNumber === 1) {
                item.classList.toggle("active", s === 1);
                item.classList.remove("completed", "processing");
                if (forkConnector) forkConnector.classList.remove("active");
                if (joinConnector) joinConnector.classList.remove("active");
            } else if (stepNumber === 2 || stepNumber === 3) {
                // Parallel phase: Agent 1 (stage 2) and Agent 2 (stage 3) execute simultaneously!
                const isStage1 = s === 1;
                const isParallel = s === 2 || s === 3;
                item.classList.toggle("completed", isStage1);
                item.classList.toggle("active", isParallel);
                if (s === 4) item.classList.remove("active", "completed", "processing");
                if (forkConnector) forkConnector.classList.add("active");
                if (joinConnector) joinConnector.classList.remove("active");
            } else if (stepNumber >= 4) {
                // Convergence phase: Final Decision
                item.classList.toggle("completed", s < 4);
                item.classList.toggle("active", s === 4);
                if (forkConnector) forkConnector.classList.add("active");
                if (joinConnector) joinConnector.classList.add("active");
            }
        });
    }

    stageItems.forEach((item) => {
        item.addEventListener("click", () => {
            const step = parseInt(item.dataset.stage, 10);
            setWorkflowStep(step);
        });
    });

    // ========================================================
    // 3. Model Performance Tabs & Bar Chart Animation
    // ========================================================
    function updateChart(tabKey) {
        if (tabKey) activeMetricTab = tabKey;
        const d = (metricsData && metricsData[activeMetricTab]) ? metricsData[activeMetricTab] : null;

        const cnnVal = (d && d.cnn != null) ? Number(d.cnn).toFixed(3) : "—";
        const rfVal = (d && d.rf != null) ? Number(d.rf).toFixed(3) : "—";
        const lrVal = (d && d.lr != null) ? Number(d.lr).toFixed(3) : "—";

        chartValCnn.textContent = cnnVal;
        chartValRf.textContent = rfVal;
        chartValLr.textContent = lrVal;

        // Animer hauteurs des barres (0 à 100%) selon les valeurs réelles
        chartBarCnn.style.height = (d && d.cnn != null) ? `${Math.min(Math.max(d.cnn * 100, 4), 100).toFixed(1)}%` : "0%";
        chartBarRf.style.height = (d && d.rf != null) ? `${Math.min(Math.max(d.rf * 100, 4), 100).toFixed(1)}%` : "0%";
        chartBarLr.style.height = (d && d.lr != null) ? `${Math.min(Math.max(d.lr * 100, 4), 100).toFixed(1)}%` : "0%";

        // Mise à jour synchronisée des 3 cartes mini "Performance Metrics"
        const metricLabels = {
            roc_auc: "ROC-AUC",
            accuracy: "Accuracy",
            recall: "Recall",
            precision: "Precision",
            f1: "F1-score"
        };
        const currentLabel = metricLabels[activeMetricTab] || activeMetricTab.toUpperCase();

        const refCnnAuc = document.getElementById("refCnnAuc");
        const refRfAuc = document.getElementById("refRfAuc");
        const refLrAuc = document.getElementById("refLrAuc");
        if (refCnnAuc) refCnnAuc.textContent = cnnVal;
        if (refRfAuc) refRfAuc.textContent = rfVal;
        if (refLrAuc) refLrAuc.textContent = lrVal;

        const refCnnLabel = document.getElementById("refCnnLabel");
        const refRfLabel = document.getElementById("refRfLabel");
        const refLrLabel = document.getElementById("refLrLabel");
        if (refCnnLabel) refCnnLabel.innerHTML = `${currentLabel}<br><small>(CNN)</small>`;
        if (refRfLabel) refRfLabel.innerHTML = `${currentLabel}<br><small>(RF)</small>`;
        if (refLrLabel) refLrLabel.innerHTML = `${currentLabel}<br><small>(LR)</small>`;
    }

    perfTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            perfTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            updateChart(tab.dataset.tab);
        });
    });

    // ========================================================
    // 4. Presets Loader & Synchronization
    // ========================================================
    async function loadPresets() {
        try {
            const res = await fetch("/api/presets");
            if (res.ok) {
                presetsData = await res.json();
            }
        } catch (e) {
            console.warn("Could not load presets from server:", e);
        }
    }

    function applyPreset(key) {
        const p = presetsData[key];
        if (!p) return;

        ageInput.value = p.age;
        bmiInput.value = p.bmi || 25.0;
        genderSelect.value = p.gender;
        smokingSelect.value = p.smoking;
        hypertensionSelect.value = String(p.hypertension);
        heartDiseaseSelect.value = String(p.heart_disease);
        glucoseInput.value = p.glucose;
        everMarriedInput.value = p.ever_married;
        workTypeInput.value = p.work_type;
        residenceInput.value = p.residence;

        // Reset user file if loading preset
        uploadedFile = null;
        imageFileInput.value = "";

        const sampleFilename = p.sample_image.split("/").pop();
        currentSampleName = sampleFilename;
        sampleImageNameInput.value = sampleFilename;

        xrayPreviewImg.src = p.sample_image;
        xrayPreviewImg.style.display = "block";
        previewEmptyState.style.display = "none";

        if (datasetSourceBanner) datasetSourceBanner.style.display = "none";
        if (btnRandomDataset) btnRandomDataset.classList.remove("active");

        setWorkflowStep(1);
    }

    presetChips.forEach((btn) => {
        btn.addEventListener("click", () => {
            if (btn.id === "btnRandomDataset") return;
            presetChips.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            applyPreset(btn.dataset.preset);
        });
    });

    // Random Case Loader from Kaggle Datasets (data/)
    async function loadRandomDatasetCase() {
        if (!btnRandomDataset) return;
        const originalText = btnRandomDataset.innerHTML;
        btnRandomDataset.innerHTML = `<span>⏳ Loading case...</span>`;
        btnRandomDataset.disabled = true;

        try {
            const res = await fetch("/api/random-case");
            if (!res.ok) throw new Error("Erreur de chargement du cas aléatoire.");
            const data = await res.json();

            // Populate clinical inputs
            ageInput.value = data.age;
            bmiInput.value = data.bmi || 24.5;
            genderSelect.value = data.gender;
            smokingSelect.value = data.smoking;
            hypertensionSelect.value = String(data.hypertension);
            heartDiseaseSelect.value = String(data.heart_disease);
            glucoseInput.value = data.glucose;
            everMarriedInput.value = data.ever_married;
            workTypeInput.value = data.work_type;
            residenceInput.value = data.residence;

            // Reset user upload
            uploadedFile = null;
            imageFileInput.value = "";

            // Update sample image token and preview
            currentSampleName = data.sample_token;
            sampleImageNameInput.value = data.sample_token;

            xrayPreviewImg.src = data.image_url;
            xrayPreviewImg.style.display = "block";
            previewEmptyState.style.display = "none";

            // Show ground truth banner
            if (datasetSourceBanner && datasetSourceText) {
                datasetSourceText.textContent = data.ground_truth_summary;
                datasetSourceBanner.style.display = "flex";
            }

            // Update active states
            presetChips.forEach((b) => b.classList.remove("active"));
            btnRandomDataset.classList.add("active");

            setWorkflowStep(1);
        } catch (err) {
            console.error("Failed to load random case:", err);
            alert("Erreur lors du tirage aléatoire dans le dataset : " + err.message);
        } finally {
            btnRandomDataset.innerHTML = originalText;
            btnRandomDataset.disabled = false;
        }
    }

    if (btnRandomDataset) {
        btnRandomDataset.addEventListener("click", loadRandomDatasetCase);
    }

    // ========================================================
    // 5. Drag-and-Drop Chest X-Ray Upload
    // ========================================================
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    dropzone.addEventListener("click", () => {
        imageFileInput.click();
    });

    imageFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        const validTypes = ["image/jpeg", "image/png", "image/jpg"];
        if (!validTypes.includes(file.type)) {
            alert("Veuillez sélectionner un fichier image valide (JPEG ou PNG).");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert("L'image dépasse la taille limite autorisée de 10 Mo.");
            return;
        }

        uploadedFile = file;
        currentSampleName = null;
        sampleImageNameInput.value = "";

        const reader = new FileReader();
        reader.onload = (e) => {
            xrayPreviewImg.src = e.target.result;
            xrayPreviewImg.style.display = "block";
            previewEmptyState.style.display = "none";
        };
        reader.readAsDataURL(file);

        // Remove active preset highlight
        presetChips.forEach((b) => b.classList.remove("active"));
        if (datasetSourceBanner) datasetSourceBanner.style.display = "none";
    }

    btnClearPreview.addEventListener("click", (e) => {
        e.stopPropagation();
        uploadedFile = null;
        currentSampleName = null;
        sampleImageNameInput.value = "";
        imageFileInput.value = "";
        xrayPreviewImg.src = "";
        xrayPreviewImg.style.display = "none";
        previewEmptyState.style.display = "flex";
        if (datasetSourceBanner) datasetSourceBanner.style.display = "none";
        if (btnRandomDataset) btnRandomDataset.classList.remove("active");
    });

    // ========================================================
    // 6. Execute Triage Assessment (Submit)
    // ========================================================
    assessmentForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. UI Loading state
        btnStartAnalysis.disabled = true;
        btnSpinner.style.display = "block";
        const btnText = btnStartAnalysis.querySelector(".btn-text");
        btnText.textContent = "Analyzing...";

        // 2. Prepare FormData
        const formData = new FormData(assessmentForm);
        if (uploadedFile) {
            formData.set("image", uploadedFile);
        } else if (currentSampleName) {
            formData.set("sample_image_name", currentSampleName);
        }

        // 3. Step progression : Step 2 (Parallel Execution of Agent 1 & Agent 2)
        setWorkflowStep(2);
        const stage1 = document.getElementById("stage1");
        const stage2 = document.getElementById("stage2");
        const stage3 = document.getElementById("stage3");
        const stage4 = document.getElementById("stage4");

        if (stage1) stage1.classList.add("completed");
        if (stage2) stage2.classList.add("processing");
        if (stage3) stage3.classList.add("processing");

        try {
            const response = await fetch("/api/triage", {
                method: "POST",
                body: formData
            });

            if (stage2) {
                stage2.classList.remove("processing");
                stage2.classList.add("completed");
            }
            if (stage3) {
                stage3.classList.remove("processing");
                stage3.classList.add("completed");
            }

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();

            // Transition vers Agent 3 : Décision SAU & Convergence (200ms)
            setWorkflowStep(4);
            if (stage4) stage4.classList.add("processing");
            await new Promise((r) => setTimeout(r, 220));
            if (stage4) {
                stage4.classList.remove("processing");
                stage4.classList.add("completed");
            }

            renderResults(data);
            addRecentAssessment(data);

        } catch (err) {
            if (stage2) stage2.classList.remove("processing");
            if (stage3) stage3.classList.remove("processing");
            if (stage4) stage4.classList.remove("processing");
            alert(`Erreur d'analyse : ${err.message}`);
            setWorkflowStep(1);
        } finally {
            btnStartAnalysis.disabled = false;
            btnSpinner.style.display = "none";
            btnText.textContent = "Start Analysis";
        }
    });

    // ========================================================
    // 7. Render Analysis & Results
    // ========================================================
    function renderResults(data) {
        const { triage, agents } = data;

        resultsEmptyState.style.display = "none";
        resultsPopulatedState.style.display = "block";

        // Triage Banner Status
        triageBanner.className = `triage-status-banner ${triage.couleur}`;
        triageLevelBadge.textContent = triage.niveau;
        triageTimeframe.textContent = triage.delai_recommande;
        triageScoreNumber.textContent = `${triage.score_urgence}%`;
        triageConduite.textContent = triage.conduite_a_tenir;

        // Agent 1 CNN
        const pImg = agents.image.risque_pneumonie;
        agent1ScoreText.textContent = `${pImg}%`;
        agent1Bar.style.width = `${pImg}%`;
        agent1Bar.className = `progress-bar-fill ${pImg >= 50 ? "red" : "teal"}`;
        agent1StatusLabel.textContent = agents.image.statut;

        // Agent 2 RF (synchronisé sur seuil_utilise OOF du modèle, par défaut 65%)
        const pClin = agents.clinique.risque_avc;
        const rfThreshPct = Math.round((agents.clinique.seuil_utilise ?? 0.65) * 100);
        agent2ScoreText.textContent = `${pClin}%`;
        agent2Bar.style.width = `${pClin}%`;
        agent2Bar.className = `progress-bar-fill ${pClin >= rfThreshPct ? "red" : (pClin >= 40 ? "amber" : "teal")}`;
        agent2StatusLabel.textContent = agents.clinique.statut;

        // Agent 3 Consensus
        agent3ScoreText.textContent = `${triage.score_urgence}%`;
        formulaChipBox.innerHTML = `<code>${agents.decision.formule}</code>`;
    }

    // ========================================================
    // 8. Recent Assessments History List
    // ========================================================
    function addRecentAssessment(data) {
        const id = "EVAL-" + Math.floor(1000 + Math.random() * 9000);
        const now = new Date();
        const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        const time = `${months[now.getMonth()]} ${now.getDate()}, ${now.getFullYear()} – ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

        const item = {
            id,
            time,
            niveau: data.triage.niveau,
            score: data.triage.score_urgence,
            couleur: data.triage.couleur,
            data
        };

        sessionAssessments.unshift(item);
        if (sessionAssessments.length > 5) {
            sessionAssessments.pop();
        }

        renderRecentList();
    }

    function renderRecentList() {
        if (sessionAssessments.length === 0) {
            recentEmptyState.style.display = "none";
            recentListContainer.style.display = "flex";
            return;
        }

        recentEmptyState.style.display = "none";
        recentListContainer.style.display = "flex";
        recentListContainer.innerHTML = "";

        sessionAssessments.forEach((item) => {
            const row = document.createElement("div");
            row.className = "recent-item-row";
            row.style.cursor = "pointer";

            let badgeClass = "analyzed";
            let badgeText = item.niveau;
            if (item.niveau === "URGENT") {
                badgeClass = "urgent";
                badgeText = "Urgent";
            } else if (item.niveau === "NON URGENT") {
                badgeClass = "completed";
                badgeText = "Completed";
            } else {
                badgeClass = "analyzed";
                badgeText = "Analyzed";
            }

            row.innerHTML = `
                <div class="recent-item-meta">
                    <svg class="recent-file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span class="recent-date-text">${item.time}</span>
                </div>
                <span class="status-pill-badge ${badgeClass}">${badgeText}</span>
            `;
            row.addEventListener("click", () => {
                renderResults(item.data);
            });
            recentListContainer.appendChild(row);
        });
    }

    // ========================================================
    // 9. Startup Initialization & Live Auto-sync
    // ========================================================
    checkApiConnection();
    loadPresets();
    updateChart("roc_auc");
    setWorkflowStep(1);

    // Synchronisation automatique en direct (mise à jour instantanée si Agent 3 ou un modèle change)
    window.addEventListener("focus", checkApiConnection);
    setInterval(checkApiConnection, 10000);
});
