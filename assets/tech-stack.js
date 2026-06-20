(function () {
  const canvas = document.getElementById("tech-stack-canvas");
  if (!canvas) return;

  const shell = canvas.closest(".tech-scene-wrap");
  const title = document.querySelector("[data-tech-title]");
  const detail = document.querySelector("[data-tech-detail]");
  const buttons = document.querySelectorAll("[data-tech-step]");
  const labels = document.querySelectorAll("[data-tech-label]");
  const metricA = document.querySelector("[data-tech-metric-a]");
  const metricB = document.querySelector("[data-tech-metric-b]");
  const metricC = document.querySelector("[data-tech-metric-c]");
  const metricLabelA = document.querySelector("[data-tech-metric-label-a]");
  const metricLabelB = document.querySelector("[data-tech-metric-label-b]");
  const metricLabelC = document.querySelector("[data-tech-metric-label-c]");

  const steps = [
    {
      title: "Feature Matrix",
      detail:
        "pandas and NumPy shape one cafeteria item-period into a 22-column feature vector. The animation shows row vectors flowing through scaled numeric signals and one-hot menu/event categories.",
      metrics: [
        ["22", "features"],
        ["10.6k", "training rows"],
        ["540", "school days"]
      ],
      focus: "features"
    },
    {
      title: "Model Ensemble",
      detail:
        "scikit-learn heads run in parallel: a ghost-risk classifier, quantile return regressors, after-school demand matching, and an anomaly detector for unusual operating days.",
      metrics: [
        ["4", "model families"],
        ["Q10-90", "return range"],
        ["0.82", "ghost risk"]
      ],
      focus: "models"
    },
    {
      title: "Safety Compiler",
      detail:
        "Predictions pass through a rule state machine before they become actions: sealed packaging, cold-chain timing, monitor status, cooler capacity, confidence, and manager approval.",
      metrics: [
        ["7", "gates"],
        ["12:12", "deploy by"],
        ["human", "approval"]
      ],
      focus: "policy"
    },
    {
      title: "Product Layer",
      detail:
        "Streamlit and Plotly render the live cafeteria workflow: scenario controls, action cards, approval toggles, and an impact receipt that only counts safe marginal recovery.",
      metrics: [
        ["175", "items capped"],
        ["31kg", "diverted"],
        ["52kg", "CO2e avoided"]
      ],
      focus: "product"
    }
  ];

  let activeStep = 0;
  let sceneApi = null;

  function writeText(index) {
    const step = steps[index];
    if (!step) return;
    activeStep = index;

    if (title) title.textContent = step.title;
    if (detail) detail.textContent = step.detail;
    if (metricA) metricA.textContent = step.metrics[0][0];
    if (metricB) metricB.textContent = step.metrics[1][0];
    if (metricC) metricC.textContent = step.metrics[2][0];
    if (metricLabelA) metricLabelA.textContent = step.metrics[0][1];
    if (metricLabelB) metricLabelB.textContent = step.metrics[1][1];
    if (metricLabelC) metricLabelC.textContent = step.metrics[2][1];

    buttons.forEach((button, buttonIndex) => {
      const isActive = buttonIndex === index;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });
    labels.forEach((label, labelIndex) => {
      label.classList.toggle("active", labelIndex === index);
    });

    if (sceneApi) sceneApi.setFocus(index);
  }

  buttons.forEach((button, index) => {
    button.addEventListener("click", () => writeText(index));
  });
  writeText(0);

  if (!window.THREE || !shell) {
    if (shell) shell.classList.add("tech-three-failed");
    return;
  }

  const THREE = window.THREE;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  const root = new THREE.Group();
  const focusGroups = [[], [], [], []];
  const pulseMeshes = [];
  const particles = [];
  const clock = new THREE.Clock();
  const pointer = { x: 0, y: 0 };
  let visible = true;
  let targetFocus = 0;
  let currentFocus = 0;

  camera.position.set(0, 2.2, 11.2);
  scene.add(root);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
  keyLight.position.set(2.8, 6, 5);
  scene.add(keyLight);

  const rimLight = new THREE.PointLight(0xb7d9ff, 1.6, 18);
  rimLight.position.set(-4.5, 2.5, 4);
  scene.add(rimLight);

  const palette = {
    paper: 0xf7f7f2,
    cool: 0xb7d9ff,
    amber: 0xe5c15a,
    dark: 0x151515,
    line: 0x6f879f
  };

  function material(color, opacity, emissive) {
    return new THREE.MeshStandardMaterial({
      color,
      emissive: emissive || 0x000000,
      emissiveIntensity: emissive ? 0.18 : 0,
      roughness: 0.54,
      metalness: 0.22,
      transparent: opacity < 1,
      opacity
    });
  }

  function addToFocus(mesh, index) {
    mesh.userData.homeScale = mesh.scale.clone();
    mesh.userData.baseOpacity = mesh.material && mesh.material.opacity ? mesh.material.opacity : 1;
    focusGroups[index].push(mesh);
    return mesh;
  }

  function box(w, h, d, color, opacity, x, y, z, focusIndex) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material(color, opacity));
    mesh.position.set(x, y, z);
    root.add(mesh);
    return addToFocus(mesh, focusIndex);
  }

  function sphere(radius, color, opacity, x, y, z, focusIndex) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 16), material(color, opacity, color));
    mesh.position.set(x, y, z);
    root.add(mesh);
    pulseMeshes.push(mesh);
    return addToFocus(mesh, focusIndex);
  }

  function line(points, color, opacity) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const mesh = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity })
    );
    root.add(mesh);
    return mesh;
  }

  function makeMatrix() {
    for (let row = 0; row < 6; row += 1) {
      for (let col = 0; col < 7; col += 1) {
        const height = 0.08 + ((row + col) % 4) * 0.035;
        const x = -4.7 + col * 0.28;
        const y = 1.15 - row * 0.3;
        const z = -0.25 + Math.sin((row + col) * 0.7) * 0.18;
        const cell = box(0.19, height, 0.19, row % 2 ? palette.cool : palette.paper, 0.72, x, y, z, 0);
        cell.userData.phase = row * 0.2 + col * 0.09;
      }
    }

    for (let col = 0; col < 7; col += 1) {
      const bar = box(0.1, 1.5 + (col % 3) * 0.22, 0.08, palette.cool, 0.34, -4.72 + col * 0.28, -1.34, 0, 0);
      bar.userData.phase = col * 0.23;
    }
  }

  function makeWeights() {
    for (let row = 0; row < 6; row += 1) {
      for (let col = 0; col < 6; col += 1) {
        const weight = box(0.08, 0.22, 0.8, (row + col) % 2 ? palette.paper : palette.amber, 0.42, -1.85 + col * 0.22, 0.92 - row * 0.29, 0, 1);
        weight.rotation.y = -0.42;
        weight.userData.phase = row * 0.18 + col * 0.08;
      }
    }

    const logits = [-0.62, -0.18, 0.28, 0.72];
    logits.forEach((y, index) => {
      const logit = box(0.92 + index * 0.14, 0.08, 0.08, palette.cool, 0.72, 0.1, y, 0.1, 1);
      logit.userData.phase = index * 0.26;
    });
  }

  function makeHeads() {
    const heads = [
      ["risk", 0.98],
      ["q", 0.34],
      ["demand", -0.32],
      ["anomaly", -0.96]
    ];
    heads.forEach((head, index) => {
      const node = sphere(0.19, index === 1 ? palette.amber : palette.cool, 0.92, 1.58, head[1], 0, 1);
      node.userData.phase = index * 0.32;
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.33, 0.01, 8, 48),
        material(palette.paper, 0.45)
      );
      ring.position.copy(node.position);
      ring.rotation.x = Math.PI / 2;
      root.add(ring);
      addToFocus(ring, 1);
    });
  }

  function makePolicy() {
    for (let index = 0; index < 7; index += 1) {
      const y = 1.2 - index * 0.38;
      const gate = box(0.94, 0.08, 0.12, index % 3 === 0 ? palette.amber : palette.paper, 0.74, 3.26, y, 0, 2);
      gate.userData.phase = index * 0.17;
    }

    const shield = new THREE.Mesh(
      new THREE.TorusGeometry(0.96, 0.018, 10, 72),
      material(palette.cool, 0.36)
    );
    shield.position.set(3.26, 0, -0.05);
    shield.scale.y = 1.42;
    root.add(shield);
    addToFocus(shield, 2);
  }

  function makeProduct() {
    const card = box(1.55, 1.95, 0.06, palette.paper, 0.92, 5.1, 0.1, 0, 3);
    card.rotation.y = -0.18;
    for (let index = 0; index < 5; index += 1) {
      const row = box(1.02 - index * 0.04, 0.055, 0.08, index === 0 ? palette.dark : palette.cool, index === 0 ? 0.95 : 0.55, 5.08, 0.78 - index * 0.34, 0.08, 3);
      row.rotation.y = -0.18;
    }
    const approval = sphere(0.14, palette.amber, 0.95, 5.8, -0.74, 0.1, 3);
    approval.scale.set(1.9, 1.9, 0.5);
  }

  makeMatrix();
  makeWeights();
  makeHeads();
  makePolicy();
  makeProduct();

  const curvePoints = [
    new THREE.Vector3(-4.7, 0.08, 0.2),
    new THREE.Vector3(-2.1, 0.12, 0.3),
    new THREE.Vector3(1.58, 0.02, 0.2),
    new THREE.Vector3(3.24, 0.04, 0.12),
    new THREE.Vector3(5.05, 0.06, 0.12)
  ];
  const dataCurve = new THREE.CatmullRomCurve3(curvePoints);
  line(dataCurve.getPoints(80), palette.line, 0.4);

  for (let index = 0; index < 32; index += 1) {
    const particle = new THREE.Mesh(
      new THREE.SphereGeometry(0.035 + (index % 3) * 0.012, 12, 8),
      material(index % 5 === 0 ? palette.amber : palette.cool, 0.88, palette.cool)
    );
    particle.userData.offset = index / 32;
    particle.userData.speed = 0.055 + (index % 6) * 0.008;
    root.add(particle);
    particles.push(particle);
  }

  function setFocus(index) {
    targetFocus = index;
    focusGroups.forEach((group, groupIndex) => {
      group.forEach((mesh) => {
        const active = groupIndex === index || (index === 3 && groupIndex === 2);
        if (mesh.material) {
          mesh.material.opacity = active ? Math.min(1, (mesh.userData.baseOpacity || 1) + 0.18) : 0.22;
          mesh.material.transparent = true;
          if (mesh.material.emissiveIntensity !== undefined) mesh.material.emissiveIntensity = active ? 0.28 : 0.03;
        }
        mesh.userData.targetScale = active ? 1.08 : 0.96;
      });
    });
  }

  function resize() {
    const width = Math.max(1, shell.clientWidth);
    const height = Math.max(1, shell.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.position.z = width < 560 ? 13.2 : width < 900 ? 12.2 : 11.2;
    camera.updateProjectionMatrix();
  }

  function animate() {
    window.requestAnimationFrame(animate);
    if (!visible) return;

    const elapsed = clock.getElapsedTime();
    currentFocus += (targetFocus - currentFocus) * 0.06;
    root.rotation.y = pointer.x * 0.1 + (currentFocus - 1.5) * 0.075;
    root.rotation.x = -0.08 + pointer.y * 0.045;
    root.position.x = -0.2 - currentFocus * 0.06;

    focusGroups.forEach((group) => {
      group.forEach((mesh) => {
        const targetScale = mesh.userData.targetScale || 1;
        const pulse = mesh.userData.phase ? Math.sin(elapsed * 2.2 + mesh.userData.phase) * 0.018 : 0;
        const base = mesh.userData.homeScale || new THREE.Vector3(1, 1, 1);
        mesh.scale.lerp(
          new THREE.Vector3(base.x * (targetScale + pulse), base.y * (targetScale + pulse), base.z * (targetScale + pulse)),
          0.08
        );
        if (mesh.rotation && mesh.geometry && mesh.geometry.type === "TorusGeometry") {
          mesh.rotation.z += 0.006;
        }
      });
    });

    pulseMeshes.forEach((mesh, index) => {
      mesh.position.z = Math.sin(elapsed * 1.6 + index) * 0.06;
    });

    particles.forEach((particle) => {
      const t = (elapsed * particle.userData.speed + particle.userData.offset) % 1;
      const point = dataCurve.getPointAt(t);
      particle.position.copy(point);
      particle.position.y += Math.sin((t + elapsed) * Math.PI * 2) * 0.08;
      particle.material.opacity = 0.2 + Math.sin(t * Math.PI) * 0.75;
    });

    renderer.render(scene, camera);
  }

  shell.classList.add("tech-three-ready");
  sceneApi = { setFocus };
  setFocus(activeStep);
  writeText(activeStep);
  resize();
  animate();

  shell.addEventListener("pointermove", (event) => {
    const rect = shell.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    pointer.y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
  });
  shell.addEventListener("pointerleave", () => {
    pointer.x = 0;
    pointer.y = 0;
  });

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(shell);

  const visibilityObserver = new IntersectionObserver(
    ([entry]) => {
      visible = entry.isIntersecting;
    },
    { threshold: 0.05 }
  );
  visibilityObserver.observe(shell);
})();
