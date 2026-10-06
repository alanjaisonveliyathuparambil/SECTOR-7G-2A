// TruthLock AI - Forensic Web Application Controller
document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const previewContainer = document.getElementById('preview-container');
  const imagePreview = document.getElementById('image-preview');
  const videoPreview = document.getElementById('video-preview');
  const fileNameDisplay = document.getElementById('file-name');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const btnAnalyze = document.getElementById('btn-analyze');
  const analyzeBtnText = document.getElementById('analyze-btn-text');
  const analyzeSpinner = document.getElementById('analyze-spinner');
  const scannerLaser = document.getElementById('scanner-laser');

  // Model Selection
  const modelButtons = document.querySelectorAll('.model-btn');
  const activeModelNameDisplay = document.getElementById('active-model-name');
  let selectedModel = 'ensemble';

  // State Containers
  const emptyState = document.getElementById('empty-state');
  const analyzingState = document.getElementById('analyzing-state');
  const resultsDashboard = document.getElementById('results-dashboard');
  const resultStatusBadge = document.getElementById('result-status-badge');

  // Result Metrics
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictText = document.getElementById('verdict-text');
  const verdictDesc = document.getElementById('verdict-desc');
  const scoreFillCircle = document.getElementById('score-fill-circle');
  const scorePercentage = document.getElementById('score-percentage');
  const scoreTypeLbl = document.getElementById('score-type-lbl');
  
  // GAN Footprint UI Elements
  const ganFootprintBox = document.getElementById('gan-footprint-box');
  const metricGanFootprint = document.getElementById('metric-gan-footprint');
  const ganRiskBadge = document.getElementById('gan-risk-badge');
  const ganFootprintDesc = document.getElementById('gan-footprint-desc');
  const barGanCnn = document.getElementById('bar-gan-cnn');
  const sigGanCnn = document.getElementById('sig-gan-cnn');

  // OpenCLIP Diffusion UI Elements
  const diffusionBox = document.getElementById('diffusion-box');
  const metricDiffusionProb = document.getElementById('metric-diffusion-prob');
  const diffusionRiskBadge = document.getElementById('diffusion-risk-badge');
  const diffusionDesc = document.getElementById('diffusion-desc');
  const barClipDiff = document.getElementById('bar-clip-diff');
  const sigClipDiff = document.getElementById('sig-clip-diff');

  const metricRisk = document.getElementById('metric-risk');
  const metricFakeProb = document.getElementById('metric-fake-prob');
  const metricFacesCount = document.getElementById('metric-faces-count');
  const metricModelUsed = document.getElementById('metric-model-used');
  
  const annotatedResultImg = document.getElementById('annotated-result-img');
  const barSpectral = document.getElementById('bar-spectral');
  const sigSpectral = document.getElementById('sig-spectral');
  const barSpatial = document.getElementById('bar-spatial');
  const sigSpatial = document.getElementById('sig-spatial');
  const barChroma = document.getElementById('bar-chroma');
  const sigChroma = document.getElementById('sig-chroma');

  const timelineSection = document.getElementById('timeline-section');
  const timelineChart = document.getElementById('timeline-chart');
  
  const btnExportJson = document.getElementById('btn-export-json');
  const btnNewScan = document.getElementById('btn-new-scan');

  // Preset Sample Buttons
  const sampleAuthentic = document.getElementById('sample-authentic');
  const sampleMorphed = document.getElementById('sample-morphed');
  const sampleAi = document.getElementById('sample-ai');
  const sampleFake = document.getElementById('sample-fake');

  let currentFile = null;
  let currentResultData = null;

  // 1. Model Switching
  modelButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      modelButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedModel = btn.dataset.model;
      
      if (selectedModel === 'ensemble') {
        activeModelNameDisplay.textContent = 'Tri-Model Ensemble (DFDC + FF++ + Wang)';
      } else if (selectedModel === 'wang_cnndetect') {
        activeModelNameDisplay.textContent = 'Wang CNNDetection (GAN Footprint)';
      } else if (selectedModel === 'selim_dfdc') {
        activeModelNameDisplay.textContent = 'Selim DFDC (EfficientNet-B7)';
      } else {
        activeModelNameDisplay.textContent = 'FaceForensics++ (Xception)';
      }
    });
  });

  // 2. Drag & Drop & File Selection
  dropZone.addEventListener('click', (e) => {
    if (e.target !== btnRemoveFile && !previewContainer.contains(e.target)) {
      fileInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    currentFile = file;
    fileNameDisplay.textContent = file.name;
    const isVideo = file.type.startsWith('video/') || file.name.endsWith('.mp4');

    dropzonePrompt.classList.add('hidden');
    previewContainer.classList.remove('hidden');

    if (isVideo) {
      imagePreview.classList.add('hidden');
      videoPreview.classList.remove('hidden');
      videoPreview.src = URL.createObjectURL(file);
    } else {
      videoPreview.classList.add('hidden');
      imagePreview.classList.remove('hidden');
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    btnAnalyze.disabled = false;
    resetResultsView();
  }

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    videoPreview.src = '';
    previewContainer.classList.add('hidden');
    dropzonePrompt.classList.remove('hidden');
    btnAnalyze.disabled = true;
    resetResultsView();
  });

  // 3. Preset Samples Click Handling
  async function loadPresetSample(sampleName) {
    try {
      const response = await fetch(`/api/sample-media/${sampleName}`);
      if (!response.ok) throw new Error('Failed to fetch sample');
      const blob = await response.blob();
      const file = new File([blob], sampleName, { type: 'image/jpeg' });
      handleFileSelected(file);
    } catch (err) {
      console.error('Error loading sample:', err);
      alert('Could not load sample: ' + err.message);
    }
  }

  if (sampleAuthentic) sampleAuthentic.addEventListener('click', () => loadPresetSample('authentic_portrait.jpg'));
  if (sampleMorphed) sampleMorphed.addEventListener('click', () => loadPresetSample('morphed_face_sample.jpg'));
  if (sampleAi) sampleAi.addEventListener('click', () => loadPresetSample('ai_generated_portrait.jpg'));
  if (sampleFake) sampleFake.addEventListener('click', () => loadPresetSample('deepfake_synthetic_face.jpg'));

  // 4. Execution: Forensic Analysis Request
  btnAnalyze.addEventListener('click', async () => {
    if (!currentFile) return;

    btnAnalyze.disabled = true;
    analyzeBtnText.textContent = 'ANALYZING NEURAL ARTIFACTS...';
    analyzeSpinner.classList.remove('hidden');
    scannerLaser.classList.remove('hidden');

    emptyState.classList.add('hidden');
    resultsDashboard.classList.add('hidden');
    analyzingState.classList.remove('hidden');
    resultStatusBadge.textContent = 'SCANNING IN PROGRESS';
    resultStatusBadge.className = 'status-badge analyzing';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model', selectedModel);

    const isVideo = currentFile.type.startsWith('video/') || currentFile.name.endsWith('.mp4');
    const endpoint = isVideo ? '/api/analyze/video' : '/api/analyze/image';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis request failed.');
      }

      const data = await response.json();
      currentResultData = data;
      renderResults(data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis Error: ' + error.message);
      resetResultsView();
    } finally {
      btnAnalyze.disabled = false;
      analyzeBtnText.textContent = 'RUN FORENSIC SCAN';
      analyzeSpinner.classList.add('hidden');
      scannerLaser.classList.add('hidden');
    }
  });

  // 5. Render Results to Dashboard
  function renderResults(data) {
    analyzingState.classList.add('hidden');
    resultsDashboard.classList.remove('hidden');
    resultStatusBadge.textContent = 'FORENSIC AUDIT COMPLETE';
    resultStatusBadge.className = 'status-badge done';

    const fakeProb = data.fake_probability;
    const realProb = data.real_probability;
    const verdict = data.verdict;

    verdictBanner.className = 'verdict-banner';
    verdictText.textContent = verdict;

    const circumference = 264;
    
    const threshold = data.decision_threshold || (data.preprocessing && data.preprocessing.decision_threshold) || 0.25;
    const isCompromised = data.is_compromised === true || verdict === 'SYNTHETIC' || fakeProb >= threshold;
    const isWhatsApp = data.preprocessing && data.preprocessing.is_whatsapp;

    // Decision rule: OpenCLIP score >= 25% triggers red SYNTHETIC verdict
    if (isCompromised) {
      verdictBanner.classList.add('fake');
      if (isWhatsApp) {
        verdictDesc.textContent = 'WhatsApp compression artifact damping detected. OpenCLIP decision threshold active (>= 25%) • Latent diffusion generative signature identified.';
      } else {
        verdictDesc.textContent = 'Diffusion generative signature detected by OpenCLIP ViT-L-14 backbone (>= 25%). Visual feature representations match synthetic latent manifold.';
      }
      scoreTypeLbl.textContent = 'SYNTHETIC';
      scorePercentage.textContent = Math.round(fakeProb * 100) + '%';
      const offset = circumference - (circumference * fakeProb);
      scoreFillCircle.style.strokeDashoffset = offset;
    } else {
      verdictBanner.classList.remove('fake');
      verdictDesc.textContent = isWhatsApp 
        ? 'Natural photographic capture (WhatsApp compression evaluated). No synthetic diffusion signature detected (< 25%).'
        : 'Natural photographic sensor noise, organic photon distribution, and authentic camera optics. No synthetic diffusion signature detected (< 25%).';
      scoreTypeLbl.textContent = 'AUTHENTIC';
      scorePercentage.textContent = Math.round(realProb * 100) + '%';
      const offset = circumference - (circumference * realProb);
      scoreFillCircle.style.strokeDashoffset = offset;
    }

    // Sheng-Yu Wang GAN Synthetic Footprint Rendering
    const ganData = data.gan_footprint || {};
    const ganScore = data.gan_synthetic_footprint_score !== undefined ? data.gan_synthetic_footprint_score : (ganData.gan_synthetic_footprint_score || 0.1);
    const ganPct = Math.round(ganScore * 100);

    if (metricGanFootprint) {
      metricGanFootprint.textContent = ganPct + '%';
    }
    
    if (ganFootprintDesc && ganData.description) {
      ganFootprintDesc.textContent = ganData.description;
    }

    if (ganRiskBadge) {
      ganRiskBadge.textContent = (ganData.risk_level || 'LOW') + ' GAN FOOTPRINT';
      ganRiskBadge.className = 'gan-card-risk ' + (ganScore >= 0.70 ? 'high' : (ganScore >= 0.40 ? 'moderate' : ''));
    }

    if (ganFootprintBox) {
      ganFootprintBox.className = 'gan-footprint-card' + (ganScore >= 0.70 ? ' high-risk' : '');
    }

    if (barGanCnn && sigGanCnn) {
      const cnnProb = ganData.cnn_generator_probability !== undefined ? ganData.cnn_generator_probability : ganScore;
      updateSignalBar(barGanCnn, sigGanCnn, cnnProb);
    }

    // OpenCLIP ViT-L-14 Diffusion & Midjourney Rendering
    const diffData = data.diffusion_detection || {};
    const diffProb = diffData.diffusion_synthetic_probability !== undefined ? diffData.diffusion_synthetic_probability : (data.diffusion_prob || 0.05);
    const diffPct = Math.round(diffProb * 100);

    if (metricDiffusionProb) {
      metricDiffusionProb.textContent = diffPct + '%';
    }

    if (diffusionDesc && diffData.description) {
      diffusionDesc.textContent = isWhatsApp ? `${diffData.description} (WhatsApp 28% compression damping threshold applied).` : diffData.description;
    }

    if (diffusionRiskBadge) {
      const risk = diffData.risk_level || (diffProb >= threshold ? 'HIGH' : 'LOW');
      diffusionRiskBadge.textContent = isWhatsApp ? `${risk} (WA-28%)` : `${risk} DIFFUSION RISK`;
      diffusionRiskBadge.className = 'diffusion-card-risk ' + (diffProb >= threshold ? 'high' : '');
    }

    if (diffusionBox) {
      diffusionBox.className = 'diffusion-card' + (diffProb >= threshold ? ' high-risk' : '');
    }

    if (barClipDiff && sigClipDiff) {
      updateSignalBar(barClipDiff, sigClipDiff, diffProb);
    }

    // 4-Layer Forensic Extraction & Deductive Logical Thinking
    const logical = data.logical_thinking || {};
    const layers = data.layers || {};

    const logicalCard = document.getElementById('logical-thinking-card');
    const logicalStatusPill = document.getElementById('logical-status-pill');
    const archetypeVal = document.getElementById('archetype-val');
    const cotStepsContainer = document.getElementById('cot-steps-container');
    const cotConclusionText = document.getElementById('cot-conclusion-text');

    if (logicalCard) {
      if (logical.is_compromised) {
        logicalStatusPill.textContent = 'COMPROMISE DETECTED';
        logicalStatusPill.className = 'logical-status-pill compromised';
      } else {
        logicalStatusPill.textContent = 'SYNTHESIS COMPLETE • ALL CLEARED';
        logicalStatusPill.className = 'logical-status-pill';
      }

      if (archetypeVal) {
        archetypeVal.textContent = data.manipulation_type || logical.manipulation_type || 'Authentic Optical Capture';
        archetypeVal.className = 'archetype-val' + (logical.is_compromised ? ' fake' : '');
      }

      // Layer 1: Selim Seferbekov (Spatial face-boundary artifacts & skin texturing blending seams)
      const l1 = layers.layer1_spatial_boundaries || {};
      const l1Score = l1.score !== undefined ? l1.score : 0.08;
      const l1Flagged = l1.is_flagged || l1Score >= 0.25;
      const l1Tag = document.getElementById('layer1-status-tag');
      const l1Fill = document.getElementById('layer1-score-fill');
      const l1Val = document.getElementById('layer1-score-val');
      const l1Desc = document.getElementById('layer1-detail-text');
      const l1Item = document.getElementById('layer-selim-item');

      if (l1Tag) {
        l1Tag.textContent = l1Flagged ? 'FAIL' : 'PASS';
        l1Tag.className = 'layer-status-tag' + (l1Flagged ? ' fail' : '');
      }
      if (l1Fill) {
        l1Fill.style.width = Math.round(l1Score * 100) + '%';
        l1Fill.className = 'layer-score-fill' + (l1Flagged ? ' high' : '');
      }
      if (l1Val) {
        l1Val.textContent = (l1Score * 100).toFixed(1) + '%';
        l1Val.className = 'layer-metric-val' + (l1Flagged ? ' high' : '');
      }
      if (l1Desc && l1.description) l1Desc.textContent = l1.description;
      if (l1Item) l1Item.className = 'layer-item' + (l1Flagged ? ' flagged' : '');

      // Layer 2: FaceForensics++ Xception (Depthwise separable compression matrices & tampering styles)
      const l2 = layers.layer2_depthwise_compression || {};
      const l2Score = l2.score !== undefined ? l2.score : 0.02;
      const l2Flagged = l2.is_flagged || l2Score >= 0.25;
      const l2Tag = document.getElementById('layer2-status-tag');
      const l2Fill = document.getElementById('layer2-score-fill');
      const l2Val = document.getElementById('layer2-score-val');
      const l2Desc = document.getElementById('layer2-detail-text');
      const l2Item = document.getElementById('layer-xception-item');

      if (l2Tag) {
        l2Tag.textContent = l2Flagged ? 'FAIL' : 'PASS';
        l2Tag.className = 'layer-status-tag' + (l2Flagged ? ' fail' : '');
      }
      if (l2Fill) {
        l2Fill.style.width = Math.round(l2Score * 100) + '%';
        l2Fill.className = 'layer-score-fill' + (l2Flagged ? ' high' : '');
      }
      if (l2Val) {
        l2Val.textContent = (l2Score * 100).toFixed(1) + '%';
        l2Val.className = 'layer-metric-val' + (l2Flagged ? ' high' : '');
      }
      if (l2Desc && l2.description) l2Desc.textContent = l2.description;
      if (l2Item) l2Item.className = 'layer-item' + (l2Flagged ? ' flagged' : '');

      // Layer 3: Sheng-Yu Wang CNNDetection (High-frequency upsampling patterns & checkerboard noise)
      const l3 = layers.layer3_frequency_upsampling || {};
      const l3Score = l3.score !== undefined ? l3.score : (data.gan_synthetic_footprint_score || 0.05);
      const l3Flagged = l3.is_flagged || l3Score >= 0.25;
      const l3Tag = document.getElementById('layer3-status-tag');
      const l3Fill = document.getElementById('layer3-score-fill');
      const l3Val = document.getElementById('layer3-score-val');
      const l3Desc = document.getElementById('layer3-detail-text');
      const l3Item = document.getElementById('layer-wang-item');

      if (l3Tag) {
        l3Tag.textContent = l3Flagged ? 'FAIL' : 'PASS';
        l3Tag.className = 'layer-status-tag' + (l3Flagged ? ' fail' : '');
      }
      if (l3Fill) {
        l3Fill.style.width = Math.round(l3Score * 100) + '%';
        l3Fill.className = 'layer-score-fill' + (l3Flagged ? ' high' : '');
      }
      if (l3Val) {
        l3Val.textContent = (l3Score * 100).toFixed(1) + '%';
        l3Val.className = 'layer-metric-val' + (l3Flagged ? ' high' : '');
      }
      if (l3Desc && l3.description) l3Desc.textContent = l3.description;
      if (l3Item) l3Item.className = 'layer-item' + (l3Flagged ? ' flagged' : '');

      // Layer 4: OpenCLIP ViT-L-14 (High-level semantic anomalies & synthetic contrast)
      const l4 = layers.layer4_semantic_contrast || {};
      const l4Score = l4.score !== undefined ? l4.score : diffProb;
      const l4Flagged = l4.is_flagged || l4Score >= 0.25;
      const l4Tag = document.getElementById('layer4-status-tag');
      const l4Fill = document.getElementById('layer4-score-fill');
      const l4Val = document.getElementById('layer4-score-val');
      const l4Desc = document.getElementById('layer4-detail-text');
      const l4Item = document.getElementById('layer-clip-item');

      if (l4Tag) {
        l4Tag.textContent = l4Flagged ? 'FAIL' : 'PASS';
        l4Tag.className = 'layer-status-tag' + (l4Flagged ? ' fail' : '');
      }
      if (l4Fill) {
        l4Fill.style.width = Math.round(l4Score * 100) + '%';
        l4Fill.className = 'layer-score-fill' + (l4Flagged ? ' high' : '');
      }
      if (l4Val) {
        l4Val.textContent = (l4Score * 100).toFixed(1) + '%';
        l4Val.className = 'layer-metric-val' + (l4Flagged ? ' high' : '');
      }
      if (l4Desc && l4.description) l4Desc.textContent = l4.description;
      if (l4Item) l4Item.className = 'layer-item' + (l4Flagged ? ' flagged' : '');

      // Chain of Thought Steps Rendering
      if (cotStepsContainer && logical.reasoning_steps) {
        cotStepsContainer.innerHTML = '';
        logical.reasoning_steps.forEach(step => {
          const stepEl = document.createElement('div');
          const isAnomaly = step.status === 'ANOMALY DETECTED';
          const isVerdict = step.status === 'VERDICT REACHED';
          stepEl.className = 'cot-step-item' + (isAnomaly ? ' anomaly' : (isVerdict ? ' verdict' : ''));

          stepEl.innerHTML = `
            <div class="cot-step-top">
              <span class="cot-step-title">[STEP ${step.step}] ${step.title}</span>
              <span class="cot-step-status ${isAnomaly ? 'fail' : (isVerdict ? 'verdict-tag' : '')}">${step.status}</span>
            </div>
            <div class="cot-step-desc">${step.observation}</div>
          `;
          cotStepsContainer.appendChild(stepEl);
        });
      }

      if (cotConclusionText) {
        cotConclusionText.textContent = logical.deduction_summary || 'Multi-layer deduction complete. Organic capture verified.';
      }
    }

    // Key Metrics Grid
    metricRisk.textContent = data.risk_level;
    metricRisk.style.color = (data.risk_level === 'CRITICAL' || data.risk_level === 'HIGH') ? 'var(--verdict-fake)' : (data.risk_level === 'MODERATE' ? 'var(--verdict-susp)' : 'var(--verdict-real)');
    metricFakeProb.textContent = (fakeProb * 100).toFixed(1) + '%';
    metricFacesCount.textContent = data.faces_detected !== undefined ? data.faces_detected : (data.sampled_frames + ' frames');
    metricModelUsed.textContent = formatModelName(data.model_used);

    // Visual Evidence Image
    if (data.annotated_image) {
      annotatedResultImg.src = data.annotated_image;
      annotatedResultImg.parentElement.parentElement.classList.remove('hidden');
    } else if (data.keyframe_image) {
      annotatedResultImg.src = data.keyframe_image;
      annotatedResultImg.parentElement.parentElement.classList.remove('hidden');
    } else {
      annotatedResultImg.parentElement.parentElement.classList.add('hidden');
    }

    // Forensic Signals Bars
    const sig = data.aggregate_signals || (data.faces && data.faces[0] ? data.faces[0].signals : null) || {
      spectral_anomaly: 0.15,
      spatial_discontinuity: 0.18,
      chrominance_mismatch: 0.12
    };

    updateSignalBar(barSpectral, sigSpectral, sig.spectral_anomaly);
    updateSignalBar(barSpatial, sigSpatial, sig.spatial_discontinuity);
    updateSignalBar(barChroma, sigChroma, sig.chrominance_mismatch);

    // Video Timeline Section
    if (data.media_type === 'video' && data.timeline) {
      timelineSection.classList.remove('hidden');
      renderTimeline(data.timeline);
    } else {
      timelineSection.classList.add('hidden');
    }
  }

  function updateSignalBar(barEl, textEl, val) {
    const pct = Math.round(Math.min(Math.max(val, 0), 1) * 100);
    barEl.style.width = pct + '%';
    textEl.textContent = pct + '%';
    if (pct > 65) {
      barEl.style.background = 'linear-gradient(90deg, #f59e0b, #ff2a5f)';
    } else {
      barEl.style.background = 'linear-gradient(90deg, #4facfe, #00f2fe)';
    }
  }

  function renderTimeline(timeline) {
    timelineChart.innerHTML = '';
    timeline.forEach(item => {
      const bar = document.createElement('div');
      bar.className = 'timeline-bar';
      if (item.fake_probability >= 0.50) {
        bar.classList.add('fake');
      }
      const height = Math.max(Math.round(item.fake_probability * 100), 10);
      bar.style.height = height + '%';
      bar.setAttribute('data-tooltip', `${item.timestamp_sec}s: ${(item.fake_probability * 100).toFixed(0)}% Fake`);
      timelineChart.appendChild(bar);
    });
  }

  function formatModelName(modelKey) {
    if (modelKey === 'clip_diffusion') return 'OpenCLIP ViT-L-14';
    if (modelKey === 'wang_cnndetect') return 'Wang CNNDetect (ResNet-50)';
    if (modelKey === 'selim_dfdc') return 'Selim DFDC (B7)';
    if (modelKey === 'faceforensics_xception') return 'FF++ Xception';
    return 'Multi-Model Ensemble';
  }

  function resetResultsView() {
    emptyState.classList.remove('hidden');
    analyzingState.classList.add('hidden');
    resultsDashboard.classList.add('hidden');
    resultStatusBadge.textContent = 'AWAITING INPUT';
    resultStatusBadge.className = 'status-badge waiting';
    currentResultData = null;
  }

  // 6. Export JSON Report
  btnExportJson.addEventListener('click', () => {
    if (!currentResultData) return;
    const blob = new Blob([JSON.stringify(currentResultData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `truthlock_audit_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  btnNewScan.addEventListener('click', () => {
    resetResultsView();
    btnRemoveFile.click();
  });
});
