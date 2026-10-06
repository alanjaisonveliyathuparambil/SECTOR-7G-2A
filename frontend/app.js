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

  sampleAuthentic.addEventListener('click', () => loadPresetSample('authentic_portrait.jpg'));
  sampleFake.addEventListener('click', () => loadPresetSample('deepfake_synthetic_face.jpg'));

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
    
    if (fakeProb >= 0.70) {
      verdictBanner.classList.add('fake');
      verdictDesc.textContent = 'High-confidence facial manipulation detected. Deep CNN activations identify generative or face-swap boundaries.';
      scoreTypeLbl.textContent = 'FAKE PROBABILITY';
      scorePercentage.textContent = Math.round(fakeProb * 100) + '%';
      const offset = circumference - (circumference * fakeProb);
      scoreFillCircle.style.strokeDashoffset = offset;
    } else if (fakeProb >= 0.45) {
      verdictBanner.classList.add('suspicious');
      verdictDesc.textContent = 'Moderate synthetic anomalies detected. Features indicate possible post-processing, blurring, or partial morphing.';
      scoreTypeLbl.textContent = 'MANIPULATION';
      scorePercentage.textContent = Math.round(fakeProb * 100) + '%';
      const offset = circumference - (circumference * fakeProb);
      scoreFillCircle.style.strokeDashoffset = offset;
    } else {
      verdictDesc.textContent = 'Natural facial landmark distribution, consistent chrominance, and organic noise profile. No synthetic signature identified.';
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
      diffusionDesc.textContent = diffData.description;
    }

    if (diffusionRiskBadge) {
      const risk = diffData.risk_level || (diffProb >= 0.65 ? 'CRITICAL' : (diffProb >= 0.40 ? 'MODERATE' : 'LOW'));
      diffusionRiskBadge.textContent = risk + ' DIFFUSION RISK';
      diffusionRiskBadge.className = 'diffusion-card-risk ' + (diffProb >= 0.65 ? 'high' : (diffProb >= 0.40 ? 'moderate' : ''));
    }

    if (diffusionBox) {
      diffusionBox.className = 'diffusion-card' + (diffProb >= 0.65 ? ' high-risk' : '');
    }

    if (barClipDiff && sigClipDiff) {
      updateSignalBar(barClipDiff, sigClipDiff, diffProb);
    }

    // Key Metrics Grid
    metricRisk.textContent = data.risk_level;
    metricRisk.style.color = data.risk_level === 'CRITICAL' ? 'var(--verdict-fake)' : (data.risk_level === 'MODERATE' ? 'var(--verdict-susp)' : 'var(--verdict-real)');
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
