const root = document.documentElement;
const reveals = document.querySelectorAll(".reveal");
const counters = document.querySelectorAll("[data-count]");
const industryButtons = document.querySelectorAll(".industry-item");
const industryCards = document.querySelectorAll(".industry-card");
const architecture = document.querySelector(".architecture");
const architectureStage = document.querySelector(".architecture-stage");
const architectureSteps = document.querySelectorAll(".arch-step");
const inference = document.querySelector(".inference-array");
const arrayStage = document.querySelector(".array-stage");
const arraySteps = document.querySelectorAll(".array-step");
const counted = new WeakSet();

let ticking = false;
let latestScroll = 0;
let industryTouched = false;

function updateScrollVars() {
  const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const ratio = Math.min(1, Math.max(0, latestScroll / max));
  root.style.setProperty("--progress", ratio.toFixed(5));
  updateArchitecture();
  updateInferenceArray();
  ticking = false;
}

function updateArchitecture() {
  if (!architecture || !architectureStage || !architectureSteps.length) return;
  const rect = architecture.getBoundingClientRect();
  const distance = Math.max(1, rect.height - window.innerHeight);
  const local = Math.min(1, Math.max(0, -rect.top / distance));
  const phase = Math.min(architectureSteps.length - 1, Math.floor(local * architectureSteps.length));

  architectureStage.style.setProperty("--arch-local", local.toFixed(4));
  architectureStage.dataset.phase = String(phase);
  architectureSteps.forEach((step, index) => {
    step.classList.toggle("active", index === phase);
  });
}

function updateInferenceArray() {
  if (!inference || !arrayStage || !arraySteps.length) return;
  const rect = inference.getBoundingClientRect();
  const distance = Math.max(1, rect.height - window.innerHeight);
  const local = Math.min(1, Math.max(0, -rect.top / distance));
  const phase = Math.min(arraySteps.length - 1, Math.floor(local * arraySteps.length));

  arrayStage.style.setProperty("--array-local", local.toFixed(4));
  arrayStage.dataset.phase = String(phase);
  arraySteps.forEach((step, index) => {
    step.classList.toggle("active", index === phase);
  });
}

function requestScrollUpdate() {
  latestScroll = window.scrollY || window.pageYOffset || 0;
  if (!ticking) {
    window.requestAnimationFrame(updateScrollVars);
    ticking = true;
  }
}

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.14, rootMargin: "0px 0px -10% 0px" }
);

reveals.forEach((element) => revealObserver.observe(element));

function animateCounter(element) {
  if (counted.has(element)) return;
  counted.add(element);
  const target = Number(element.dataset.count || 0);
  const start = performance.now();
  const duration = 1200;

  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    element.textContent = String(Math.round(target * eased));
    if (t < 1) window.requestAnimationFrame(frame);
  }

  window.requestAnimationFrame(frame);
}

const counterObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) animateCounter(entry.target);
    });
  },
  { threshold: 0.55 }
);

counters.forEach((element) => counterObserver.observe(element));

function activateIndustry(index) {
  industryButtons.forEach((button, buttonIndex) => {
    button.classList.toggle("active", buttonIndex === index);
  });
  industryCards.forEach((card, cardIndex) => {
    card.classList.toggle("active", cardIndex === index);
  });
}

industryButtons.forEach((button, index) => {
  const selectPanel = () => {
    industryTouched = true;
    activateIndustry(index);
  };
  button.addEventListener("mouseenter", selectPanel);
  button.addEventListener("focus", selectPanel);
  button.addEventListener("click", selectPanel);
});

let autoIndustry = 0;
window.setInterval(() => {
  const panel = document.querySelector(".industry-panel");
  if (!panel || document.hidden || industryTouched) return;
  const rect = panel.getBoundingClientRect();
  const visible = rect.top < window.innerHeight * 0.72 && rect.bottom > window.innerHeight * 0.28;
  if (!visible) return;
  autoIndustry = (autoIndustry + 1) % industryCards.length;
  activateIndustry(autoIndustry);
}, 3200);

window.addEventListener("scroll", requestScrollUpdate, { passive: true });
window.addEventListener("resize", requestScrollUpdate);
requestScrollUpdate();
