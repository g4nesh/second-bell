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
        "pandas and NumPy convert one cafeteria item-period into a compact 22-column feature vector. The scene shows those normalized signals as a thin matrix ribbon, not raw private student data.",
      metrics: [
        ["22", "features"],
        ["10.6k", "training rows"],
        ["540", "school days"]
      ]
    },
    {
      title: "Model Ensemble",
      detail:
        "Four scikit-learn heads read the same feature vector: ghost-risk classification, quantile return ranges, after-school demand matching, and unusual-day anomaly review.",
      metrics: [
        ["4", "model heads"],
        ["Q10-90", "return range"],
        ["0.82", "ghost risk"]
      ]
    },
    {
      title: "Safety Compiler",
      detail:
        "Predictions are filtered through policy gates before they become actions: sealed packaging, monitor status, cooler capacity, cold-chain timing, confidence, and manager approval.",
      metrics: [
        ["7", "gates"],
        ["12:12", "deploy by"],
        ["human", "approval"]
      ]
    },
    {
      title: "Product Layer",
      detail:
        "Streamlit and Plotly turn the model output into the live workflow: scenario controls, action cards, approval toggles, and a capped impact receipt.",
      metrics: [
        ["175", "items capped"],
        ["31kg", "diverted"],
        ["52kg", "CO2e avoided"]
      ]
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
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  const root = new THREE.Group();
  const focusGroups = [[], [], [], []];
  const animated = [];
  const particles = [];
  const clock = new THREE.Clock();
  const pointer = { x: 0, y: 0 };
  let visible = true;
  let targetFocus = 0;
  let currentFocus = 0;
  let dataCurve = null;

  camera.position.set(0, 1.15, 10.4);
  scene.add(root);
  scene.add(new THREE.AmbientLight(0xffffff, 0.72));

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(-2, 4, 5);
  scene.add(keyLight);

  const rimLight = new THREE.PointLight(0xb7d9ff, 1.15, 18);
  rimLight.position.set(2.2, 2.6, 4.4);
  scene.add(rimLight);

  const palette = {
    paper: 0xf7f7f2,
    dim: 0x9c9c96,
    cool: 0xb7d9ff,
    amber: 0xe5c15a,
    graphite: 0x1c1f20,
    line: 0xb7d9ff
  };

  function standard(color, opacity, emissive) {
    return new THREE.MeshStandardMaterial({
      color,
      emissive: emissive || 0x000000,
      emissiveIntensity: emissive ? 0.12 : 0,
      roughness: 0.5,
      metalness: 0.12,
      transparent: opacity < 1,
      opacity,
      depthWrite: opacity > 0.36
    });
  }

  function basic(color, opacity) {
    return new THREE.MeshBasicMaterial({
      color,
      transparent: opacity < 1,
      opacity,
      depthWrite: false
    });
  }

  function register(mesh, index) {
    mesh.userData.homeScale = mesh.scale.clone();
    mesh.userData.baseOpacity = mesh.material && mesh.material.opacity ? mesh.material.opacity : 1;
    mesh.userData.targetScale = 1;
    if (typeof index === "number") focusGroups[index].push(mesh);
    root.add(mesh);
    return mesh;
  }

  function makeTextSprite(text, x, y, z, scale, index) {
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 520;
    labelCanvas.height = 112;
    const ctx = labelCanvas.getContext("2d");
    ctx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
    ctx.fillStyle = "rgba(247,247,242,0.08)";
    roundRect(ctx, 4, 12, 512, 86, 20);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.stroke();
    ctx.fillStyle = "rgba(247,247,242,0.76)";
    ctx.font = "700 28px Inter, Arial, sans-serif";
    ctx.letterSpacing = "2px";
    ctx.fillText(text.toUpperCase(), 32, 66);

    const texture = new THREE.CanvasTexture(labelCanvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.8 }));
    sprite.position.set(x, y, z);
    sprite.scale.set(scale * 4.65, scale, 1);
    return register(sprite, index);
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
  }

  function panel(width, height, x, y, z, index, color, opacity) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height, 16, 10), standard(color || palette.paper, opacity || 0.16));
    mesh.position.set(x, y, z);
    return register(mesh, index);
  }

  function node(radius, x, y, z, index, color) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 40, 24), standard(color || palette.cool, 0.78, color || palette.cool));
    mesh.position.set(x, y, z);
    animated.push(mesh);
    return register(mesh, index);
  }

  function tube(points, radius, color, opacity, index) {
    const curve = new THREE.CatmullRomCurve3(points);
    const mesh = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 80, radius, 12, false),
      standard(color || palette.line, opacity || 0.52, color || palette.line)
    );
    register(mesh, index);
    return { mesh, curve };
  }

  function buildFeatureMatrix() {
    makeTextSprite("feature vector", -4.75, 1.95, 0.08, 0.34, 0);
    const surface = panel(1.85, 2.45, -4.72, -0.2, -0.1, 0, palette.paper, 0.1);
    surface.rotation.y = 0.08;

    for (let row = 0; row < 7; row += 1) {
      const y = 0.86 - row * 0.28;
      const width = 1.34 + Math.sin(row * 0.7) * 0.18;
      const strand = panel(width, 0.028, -4.72, y, 0.08 + row * 0.006, 0, row % 2 ? palette.cool : palette.paper, 0.78);
      strand.userData.phase = row * 0.22;
      for (let col = 0; col < 5; col += 1) {
        const dot = node(0.026 + ((row + col) % 2) * 0.01, -5.35 + col * 0.28, y + Math.sin(col) * 0.018, 0.16, 0, row % 3 ? palette.paper : palette.cool);
        dot.userData.phase = row * 0.18 + col * 0.12;
      }
    }

    for (let col = 0; col < 8; col += 1) {
      const guide = panel(0.026, 1.55 + (col % 2) * 0.18, -5.42 + col * 0.2, -1.14, 0, 0, palette.cool, 0.18);
      guide.userData.phase = col * 0.1;
    }
  }

  function buildModelLayer() {
    makeTextSprite("ensemble heads", -1.52, 1.7, 0.08, 0.32, 1);
    const weightSurface = panel(1.8, 1.82, -1.84, -0.06, -0.05, 1, palette.amber, 0.1);
    weightSurface.rotation.y = -0.18;

    for (let row = 0; row < 7; row += 1) {
      const y = 0.75 - row * 0.25;
      const strand = panel(1.46, 0.036, -1.84, y, 0.1, 1, row % 2 ? palette.amber : palette.paper, 0.46);
      strand.rotation.y = -0.18;
      strand.userData.phase = row * 0.14;
    }

    const headY = [0.82, 0.28, -0.28, -0.84];
    headY.forEach((y, index) => {
      const head = node(0.15, 0.25, y, 0.14, 1, index === 1 ? palette.amber : palette.cool);
      const halo = new THREE.Mesh(new THREE.TorusGeometry(0.31, 0.008, 12, 72), basic(index === 1 ? palette.amber : palette.cool, 0.38));
      halo.position.copy(head.position);
      halo.rotation.x = Math.PI / 2;
      halo.userData.phase = index * 0.2;
      animated.push(halo);
      register(halo, 1);
    });
  }

  function buildPolicyLayer() {
    makeTextSprite("safety compiler", 2.4, 1.58, 0.12, 0.32, 2);
    const shellRing = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.014, 16, 120), basic(palette.cool, 0.44));
    shellRing.position.set(2.4, -0.08, 0.1);
    shellRing.scale.set(1.08, 1.48, 1);
    shellRing.rotation.x = 0.08;
    register(shellRing, 2);
    animated.push(shellRing);

    for (let index = 0; index < 7; index += 1) {
      const y = 0.76 - index * 0.25;
      const gate = panel(1.05 - (index % 3) * 0.11, 0.032, 2.4, y, 0.18, 2, index % 3 === 0 ? palette.amber : palette.paper, 0.5);
      gate.userData.phase = index * 0.13;
    }
  }

  function buildProductLayer() {
    makeTextSprite("action receipt", 4.82, 1.78, 0.08, 0.32, 3);
    const card = panel(1.48, 2.02, 4.84, -0.02, 0, 3, palette.paper, 0.24);
    card.rotation.y = -0.12;
    for (let index = 0; index < 5; index += 1) {
      const row = panel(0.95 - index * 0.04, 0.038, 4.82, 0.72 - index * 0.32, 0.15, 3, index === 0 ? palette.paper : palette.cool, index === 0 ? 0.66 : 0.34);
      row.rotation.y = -0.12;
    }
    node(0.11, 5.52, -0.72, 0.16, 3, palette.amber).scale.set(1.5, 1.5, 0.8);
  }

  buildFeatureMatrix();
  buildModelLayer();
  buildPolicyLayer();
  buildProductLayer();

  const pathPoints = [
    new THREE.Vector3(-4.72, -0.05, 0.28),
    new THREE.Vector3(-2.95, -0.02, 0.3),
    new THREE.Vector3(-1.24, -0.08, 0.3),
    new THREE.Vector3(0.45, -0.02, 0.27),
    new THREE.Vector3(2.35, -0.04, 0.28),
    new THREE.Vector3(4.75, -0.05, 0.26)
  ];
  const tubeResult = tube(pathPoints, 0.012, palette.line, 0.34, null);
  dataCurve = tubeResult.curve;

  const guideCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-4.72, 0.34, 0.08),
    new THREE.Vector3(-2.2, 0.44, 0.06),
    new THREE.Vector3(0.28, 0.22, 0.06),
    new THREE.Vector3(2.45, 0.26, 0.08),
    new THREE.Vector3(4.84, 0.28, 0.08)
  ]);
  register(
    new THREE.Mesh(new THREE.TubeGeometry(guideCurve, 80, 0.006, 8, false), basic(palette.paper, 0.12)),
    null
  );

  for (let index = 0; index < 42; index += 1) {
    const particle = new THREE.Mesh(
      new THREE.SphereGeometry(0.028 + (index % 4) * 0.004, 16, 10),
      standard(index % 7 === 0 ? palette.amber : palette.cool, 0.82, palette.cool)
    );
    particle.userData.offset = index / 42;
    particle.userData.speed = 0.045 + (index % 8) * 0.004;
    root.add(particle);
    particles.push(particle);
  }

  function setFocus(index) {
    targetFocus = index;
    focusGroups.forEach((group, groupIndex) => {
      group.forEach((mesh) => {
        const active = groupIndex === index || (index === 3 && groupIndex === 2);
        if (mesh.material) {
          mesh.material.opacity = active ? Math.min(0.94, (mesh.userData.baseOpacity || 1) + 0.2) : Math.min(0.18, mesh.userData.baseOpacity || 0.18);
          mesh.material.transparent = true;
          if (mesh.material.emissiveIntensity !== undefined) mesh.material.emissiveIntensity = active ? 0.2 : 0.02;
        }
        mesh.userData.targetScale = active ? 1.06 : 0.96;
      });
    });
  }

  function resize() {
    const width = Math.max(1, shell.clientWidth);
    const height = Math.max(1, shell.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.position.z = width < 560 ? 13.2 : width < 900 ? 12.9 : 12.2;
    camera.position.y = width < 560 ? 1.42 : 1.08;
    const sceneScale = width < 560 ? 0.62 : width < 900 ? 0.78 : 0.86;
    root.scale.setScalar(sceneScale);
    camera.updateProjectionMatrix();
  }

  function animate() {
    window.requestAnimationFrame(animate);
    if (!visible) return;

    const elapsed = clock.getElapsedTime();
    currentFocus += (targetFocus - currentFocus) * 0.055;
    root.rotation.y = pointer.x * 0.055 + (currentFocus - 1.5) * 0.045;
    root.rotation.x = -0.045 + pointer.y * 0.03;
    root.position.x = -0.12 - currentFocus * 0.04;
    root.position.y = Math.sin(elapsed * 0.45) * 0.025;

    focusGroups.forEach((group) => {
      group.forEach((mesh) => {
        const targetScale = mesh.userData.targetScale || 1;
        const pulse = mesh.userData.phase ? Math.sin(elapsed * 1.8 + mesh.userData.phase) * 0.012 : 0;
        const base = mesh.userData.homeScale || new THREE.Vector3(1, 1, 1);
        mesh.scale.lerp(
          new THREE.Vector3(base.x * (targetScale + pulse), base.y * (targetScale + pulse), base.z * (targetScale + pulse)),
          0.08
        );
      });
    });

    animated.forEach((mesh, index) => {
      if (mesh.geometry && mesh.geometry.type === "TorusGeometry") mesh.rotation.z += 0.004 + index * 0.00008;
      mesh.position.z += Math.sin(elapsed * 1.2 + index) * 0.0008;
    });

    particles.forEach((particle) => {
      const t = (elapsed * particle.userData.speed + particle.userData.offset) % 1;
      const point = dataCurve.getPointAt(t);
      particle.position.copy(point);
      particle.position.y += Math.sin((t + elapsed * 0.28) * Math.PI * 2) * 0.035;
      particle.material.opacity = 0.18 + Math.sin(t * Math.PI) * 0.64;
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
