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

    // Chart Data Models (Reference evaluation metrics)
    const metricsData = {
        roc_auc: { cnn: 0.965, rf: 0.825, lr: 0.821 },
        precision: { cnn: 0.824, rf: 0.190, lr: 0.910 },
        recall: { cnn: 0.985, rf: 0.680, lr: 0.713 },
        f1: { cnn: 0.897, rf: 0.297, lr: 0.800 }
    };

    // ========================================================
    // 1. API Health Check & Metrics Initialization
    // ========================================================
    async function checkApiConnection() {
        try {
            const res = await fetch("/api/metrics");
            if (res.ok) {
                const data = await res.json();
                apiStatusDot.className = "status-dot green";
                apiStatusText.textContent = "FastAPI · Connected";

                // Update Reference Cards dynamically if available
                if (data.image && data.image.roc_auc) {
                    document.getElementById("refCnnAuc").textContent = data.image.roc_auc.toFixed(3);
                }
                if (data.clinique && data.clinique.roc_auc) {
                    document.getElementById("refRfAuc").textContent = data.clinique.roc_auc.toFixed(3);
                }
                if (data.decision && data.decision.roc_auc) {
                    document.getElementById("refLrAuc").textContent = data.decision.roc_auc.toFixed(3);
                }
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
    // 2. Interactive Workflow Stages
    // ========================================================
    function setWorkflowStep(stepNumber) {
        currentWorkflowStep = stepNumber;
        workflowStepText.textContent = `Step ${stepNumber} of 5`;

        stepDots.forEach((dot, idx) => {
            dot.classList.remove("active", "completed");
            if (idx + 1 === stepNumber) {
                dot.classList.add("active");
            } else if (idx + 1 < stepNumber) {
                dot.classList.add("completed");
            }
        });

        stageItems.forEach((item) => {
            const s = parseInt(item.dataset.stage, 10);
            item.classList.toggle("active", s === stepNumber);
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
        const d = metricsData[tabKey] || metricsData.roc_auc;

        chartValCnn.textContent = d.cnn.toFixed(3);
        chartValRf.textContent = d.rf.toFixed(3);
        chartValLr.textContent = d.lr.toFixed(3);

        // Animate heights (0 to 100%)
        chartBarCnn.style.height = `${(d.cnn * 100).toFixed(1)}%`;
        chartBarRf.style.height = `${(d.rf * 100).toFixed(1)}%`;
        chartBarLr.style.height = `${(d.lr * 100).toFixed(1)}%`;
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

        setWorkflowStep(1);
    }

    presetChips.forEach((btn) => {
        btn.addEventListener("click", () => {
            presetChips.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            applyPreset(btn.dataset.preset);
        });
    });

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

        // 3. Step progression : Step 2 (CNN is analyzing the image)
        setWorkflowStep(2);
        const stage2 = document.getElementById("stage2");
        const stage3 = document.getElementById("stage3");
        const stage4 = document.getElementById("stage4");
        const stage5 = document.getElementById("stage5");

        stage2.classList.add("processing");

        try {
            const response = await fetch("/api/triage", {
                method: "POST",
                body: formData
            });

            stage2.classList.remove("processing");

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();

            // Cascade fluide et rapide sur les agents suivants (150ms chacun)
            setWorkflowStep(3);
            stage3.classList.add("processing");
            await new Promise((r) => setTimeout(r, 150));
            stage3.classList.remove("processing");

            setWorkflowStep(4);
            stage4.classList.add("processing");
            await new Promise((r) => setTimeout(r, 150));
            stage4.classList.remove("processing");

            setWorkflowStep(5);
            stage5.classList.add("processing");
            setTimeout(() => stage5.classList.remove("processing"), 300);

            renderResults(data);
            addRecentAssessment(data);

        } catch (err) {
            stage2.classList.remove("processing");
            stage3.classList.remove("processing");
            stage4.classList.remove("processing");
            alert(`Erreur d'analyse : ${err.message}`);
            setWorkflowStep(1);
        } finally {
            btnStartAnalysis.disabled = false;
            btnSpinner.style.display = "none";
            btnText.textContent = "Start analysis";
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

        // Agent 2 RF
        const pClin = agents.clinique.risque_avc;
        agent2ScoreText.textContent = `${pClin}%`;
        agent2Bar.style.width = `${pClin}%`;
        agent2Bar.className = `progress-bar-fill ${pClin >= 50 ? "red" : (pClin >= 25 ? "amber" : "teal")}`;
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
        const time = new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });

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
            recentEmptyState.style.display = "flex";
            recentListContainer.style.display = "none";
            return;
        }

        recentEmptyState.style.display = "none";
        recentListContainer.style.display = "flex";
        recentListContainer.innerHTML = "";

        sessionAssessments.forEach((item) => {
            const row = document.createElement("div");
            row.className = "recent-item";
            row.innerHTML = `
                <div class="recent-patient-meta">
                    <span class="recent-id">${item.id}</span>
                    <span class="recent-time">${item.time} · Score: ${item.score}%</span>
                </div>
                <span class="recent-badge-pill ${item.couleur}">${item.niveau}</span>
            `;
            row.addEventListener("click", () => {
                renderResults(item.data);
            });
            recentListContainer.appendChild(row);
        });
    }

    // ========================================================
    // 9. Startup Initialization
    // ========================================================
    checkApiConnection();
    loadPresets();
    updateChart("roc_auc");
    setWorkflowStep(1);
});
