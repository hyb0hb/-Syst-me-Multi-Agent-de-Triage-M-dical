# Guide de Soutenance & Démonstration Orale
## Système Multi-Agent de Triage Médical (MedTriage AI)

Ce guide est conçu pour vous accompagner pas-à-pas lors de la présentation orale devant le jury ou le professeur. Il structure votre discours en 10 minutes et vous donne les réponses scientifiques exactes aux questions pièges.

---

## 1. Déroulé Idéal de la Présentation (10 minutes)

| Temps | Partie | Message clé à faire passer |
|---|---|---|
| **00:00 – 02:00** | **Problématique & Enjeux** | L'engorgement des urgences et l'asynchronisme des données (imagerie vs constantes cliniques). |
| **02:00 – 04:30** | **Architecture Multi-Agent** | 2 spécialistes autonomes (Vision CNN + Risque Clinique RF) arbitrés par un superviseur (Régression Logistique + Principe de Précaution SAU). |
| **04:30 – 07:00** | **Démonstration Live** | Passage en revue des 4 cas pivots sur l'interface web (http://localhost:8000). |
| **07:00 – 08:30** | **Démarche Critique & Audit** | Détection et résolution des anomalies (biais de prévalence, sous-triage unilatéral, data leakage). |
| **08:30 – 10:00** | **Conclusion & XAI** | Respect à 100% du cahier des charges, perspectives (MIMIC-IV, Grad-CAM). |

---

## 2. Le Pitch d'Introduction (2 minutes)

> *"Monsieur / Madame le membre du jury,  
> Les Services d'Accueil des Urgences sont confrontés à un défi critique : orienter rapidement les patients arrivant avec des tableaux cliniques hétérogènes.  
> Un patient qui arrive présente deux flux d'informations distincts :  
> 1. Des données cliniques immédiates (âge, tension, glycémie, antécédents).  
> 2. Une radiographie thoracique standard réalisée au lit du patient.  
> 
> Plutôt qu'un modèle 'boîte noire' monolithique, nous avons conçu un **Système Multi-Agent coopératif** :  
> - **L'Agent 1 (Vision)** se concentre sur l'analyse de l'imagerie thoracique par Transfer Learning MobileNetV2 pour estimer la probabilité d'infection respiratoire aiguë.  
> - **L'Agent 2 (Clinique)** évalue le terrain de vulnérabilité neuro-vasculaire par Random Forest.  
> - **L'Agent 3 (Consensus)** orchestre ces deux expertises via une régression logistique calibrée, couplée à un **principe de précaution clinique** inspiré de l'échelle de Manchester, garantissant qu'aucune détresse vitale unilatérale ne soit sous-évaluée.  
> 
> Passons sans attendre à la démonstration concrète sur notre interface de triage."*

---

## 3. Scénario de Démonstration Live (4 cas pivots)

Ouvrez **http://localhost:8000** devant le jury et cliquez successivement sur les 4 presets :

### Cas 1 : Détresse Respiratoire Aiguë (Pneumonie seule)
* **Profil :** Homme de 54 ans, radio avec foyer infectieux alvéolaire (`pneumonia.jpg`), constantes normales.
* **Résultat affiché :**
  * Agent Vision = **98.9%** (Opacités suspectes)
  * Agent Clinique = faible / modéré
  * **Triage Final = URGENT (< 15 min)** avec score d'urgence à **98.9%**.
* **Ce qu'il faut dire au jury :**
  > *"Remarquez ici l'application du principe de précaution clinique : même si le risque neuro-vasculaire est faible, l'infection pulmonaire est critique. Le système ne dilue pas l'urgence et oriente immédiatement le patient en filière prioritaire URGENT."*

### Cas 2 : Suspicion AVC Ischémique (Clinique seule)
* **Profil :** Femme de 72 ans, antécédents d'HTA et cardiopathie, glycémie à 215 mg/dL, radio pulmonaire normale (`normal.jpg`).
* **Résultat affiché :**
  * Agent Vision = **0.1%** (Poumons sains)
  * Agent Clinique = **Risque neuro-vasculaire élevé**
  * **Triage Final = URGENT (< 15 min)**.
* **Ce qu'il faut dire :**
  > *"Ce cas valide que l'Agent Clinique n'est pas écrasé par la radio saine. Le terrain à haut risque impose un bilan neuro-vasculaire d'urgence dans la fenêtre de reperfusion."*

### Cas 3 : Urgence Vitale Mixte (Choc & Comorbidités)
* **Profil :** Patient de 79 ans, foyer pulmonaire ET diabète/HTA décompensés.
* **Résultat :** Les deux filières en alerte $\to$ **Score global > 95% $\to$ URGENT immédiat**.
* **Ce qu'il faut dire :**
  > *"Ici, la régression logistique démontre tout l'intérêt de la fusion : les deux agents spécialistes confirment la gravité, le protocole déclenche une réanimation conjointe cardio-respiratoire."*

### Cas 4 : Contrôle de Routine (Patient ambulatoire)
* **Profil :** Jeune adulte de 28 ans, radio saine, glycémie et IMC normaux.
* **Résultat :** **NON URGENT (Filière ambulatoire standard)**, score < 10%.

---

## 4. Réponses aux Questions Pièges du Jury

### Question 1 : *"Pourquoi n'avoir pas fusionné les données au niveau des caractéristiques (Early Fusion) plutôt que par Late Stacking ?"*
> **Réponse modèle :**  
> *"C'est un choix d'intégrité scientifique fondamental. Les deux jeux de données imposés par le cahier des charges proviennent de cohortes totalement indépendantes : des enfants pédiatriques de 1 à 5 ans pour la radio (Guangzhou) et des adultes/seniors pour l'AVC. Il n'existe aucun patient commun aux deux tables.  
> Tenter une 'Early Fusion' ou concaténer des vecteurs de features de patients différents aurait constitué une aberration biologique. Nous avons donc opté pour un **Late Stacking Orchestrator**, où chaque agent apprend son domaine d'expertise sur sa cohorte, et l'arbitre modélise une règle de priorité clinique. Pour une Early Fusion, il faudrait une base appariée comme MIMIC-CXR couplée à MIMIC-IV."*

---

### Question 2 : *"Pourquoi une Régression Logistique pour l'Agent 3 plutôt qu'un Deep Neural Network ?"*
> **Réponse modèle :**  
> *"Pour trois raisons majeures :  
> 1. **L'explicabilité médicale et la transparence :** En santé (et conformément au futur règlement européen AI Act), une décision de triage doit être justifiable. La régression logistique fournit des coefficients lisibles ($z = 2{,}99 \cdot P_{\text{img}} + 4{,}24 \cdot P_{\text{clin}} - 3{,}31$).  
> 2. **La dimensionnalité :** L'Agent 3 ne reçoit en entrée que deux variables scalaires ($P_{\text{pneumonie}}$ et $P_{\text{AVC}}$). Un réseau de neurones multicouches sur deux entrées souffrirait de surapprentissage sans apporter aucun gain de performance.  
> 3. **L'auditabilité :** Le médecin peut visualiser la contribution exacte de chaque modalité."*

---

### Question 3 : *"L'Agent 2 prédit-il vraiment un AVC aigu en temps réel ?"*
> **Réponse modèle :**  
> *"Non, et nous l'avons explicitement clarifié dans le rapport technique. Le Stroke Prediction Dataset répertorie des facteurs de risque et des antécédents médicaux (âge, HTA, glycémie, tabac). L'Agent 2 évalue donc un **terrain de vulnérabilité neuro-vasculaire**. Le diagnostic d'AVC aigu in vivo nécessite un angioscanner ou une IRM cérébrale en urgence. En phase de triage aux urgences, ce score sert d'alerte pour accélérer l'accès à ce scanner."*

---

### Question 4 : *"Comment avez-vous résolu le problème de déséquilibre des classes ?"*
> **Réponse modèle :**  
> *"Nous avons appliqué trois mécanismes complémentaires :  
> 1. Pour l'Agent 1 (CNN), des poids de classes calculés de manière inversement proportionnelle à la fréquence ($w_c = \frac{N}{2 \cdot N_c}$).  
> 2. Pour l'Agent 2 (Random Forest), le paramètre `class_weight='balanced_subsample'` qui rééquilibre chaque arbre sur son sous-échantillon bootstrap, permettant d'atteindre 80% de rappel sur les AVC.  
> 3. Pour l'Agent 3 (Consensus), l'apprentissage sur une grille équilibrée en **4 quadrants cliniques**, empêchant la pneumonie (très fréquente) d'écraser l'AVC (rare)."*

---

### Question 5 : *"Quelle est votre plus grande fierté technique sur ce projet ?"*
> **Réponse modèle :**  
> *"C'est d'avoir mené un véritable audit technique critique plutôt que de nous contenter d'un modèle naïf. Nous avons identifié qu'une addition logit brute bloquait une pneumonie isolée à 42% (sous-triage dangereux), et nous avons intégré la clause de sauvegarde clinique SAU ($P_{\text{triage}} = \max(P_{\text{composite}}, P_{\text{img}}, P_{\text{clin}})$) validée par 15 tests unitaires automatisés."*

---

## 5. Synthèse des Chiffres Clés à Retenir Par Cœur

* **Agent 1 (MobileNetV2) :** ROC-AUC = **0.965** | Rappel Pneumonie = **98.5%** | Test set = 624 radios.
* **Agent 2 (Random Forest) :** 700 arbres | ROC-AUC = **0.846** | Rappel AVC = **80.0%** | Test set = 1 022 patients.
* **Agent 3 (Régression Logistique) :** ROC-AUC = **0.879** | Précision = **92.4%** | Coefficients : Image +2.99, Clinique +4.24, Biais -3.31.
* **Banc de Tests :** **15/15 tests réussis** en 3.6 secondes via `unittest` et `TestClient`.

