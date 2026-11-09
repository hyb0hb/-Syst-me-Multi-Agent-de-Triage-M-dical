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

    // Chart Data Models (Reference evaluation metrics from training)
    const metricsData = {
        roc_auc: { cnn: 0.965, rf: 0.846, lr: 0.879 },
        precision: { cnn: 0.824, rf: 0.151, lr: 0.924 },
        recall: { cnn: 0.985, rf: 0.800, lr: 0.685 },
        f1: { cnn: 0.897, rf: 0.254, lr: 0.787 }
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

                // Update Reference Cards & Chart dynamically
                if (data.image && data.image.roc_auc) {
                    const cnnVal = parseFloat(data.image.roc_auc);
                    document.getElementById("refCnnAuc").textContent = cnnVal.toFixed(3);
                    metricsData.roc_auc.cnn = cnnVal;
                }
                if (data.clinique && data.clinique.roc_auc) {
                    const rfVal = parseFloat(data.clinique.roc_auc);
                    document.getElementById("refRfAuc").textContent = rfVal.toFixed(3);
                    metricsData.roc_auc.rf = rfVal;
                }
                if (data.decision && data.decision.roc_auc) {
                    const lrVal = parseFloat(data.decision.roc_auc);
                    document.getElementById("refLrAuc").textContent = lrVal.toFixed(3);
                    metricsData.roc_auc.lr = lrVal;
                }
                updateChart("roc_auc");
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

        // 3. Step progression : Step 2 (CNN is analyzing the image)
        setWorkflowStep(2);
        const stage2 = document.getElementById("stage2");
        const stage3 = document.getElementById("stage3");
        const stage4 = document.getElementById("stage4");

        if (stage2) stage2.classList.add("processing");

        try {
            const response = await fetch("/api/triage", {
                method: "POST",
                body: formData
            });

            if (stage2) stage2.classList.remove("processing");

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();

            // Cascade fluide et rapide sur les agents suivants (150ms chacun)
            setWorkflowStep(3);
            if (stage3) stage3.classList.add("processing");
            await new Promise((r) => setTimeout(r, 150));
            if (stage3) stage3.classList.remove("processing");

            setWorkflowStep(4);
            if (stage4) stage4.classList.add("processing");
            await new Promise((r) => setTimeout(r, 150));
            if (stage4) stage4.classList.remove("processing");

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
    // 9. Startup Initialization
    // ========================================================
    checkApiConnection();
    loadPresets();
    updateChart("roc_auc");
    setWorkflowStep(1);
});
