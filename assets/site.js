const root = document.documentElement;
const reveals = document.querySelectorAll(".reveal");
const counters = document.querySelectorAll("[data-count]");
const industryButtons = document.querySelectorAll(".industry-item");
const industryCards = document.querySelectorAll(".industry-card");
const counted = new WeakSet();

let ticking = false;
let latestScroll = 0;

function updateScrollVars() {
  const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const ratio = Math.min(1, Math.max(0, latestScroll / max));
  root.style.setProperty("--progress", ratio.toFixed(5));
  ticking = false;
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
  button.addEventListener("mouseenter", () => activateIndustry(index));
  button.addEventListener("focus", () => activateIndustry(index));
  button.addEventListener("click", () => activateIndustry(index));
});

let autoIndustry = 0;
window.setInterval(() => {
  const panel = document.querySelector(".industry-panel");
  if (!panel || document.hidden) return;
  const rect = panel.getBoundingClientRect();
  const visible = rect.top < window.innerHeight * 0.72 && rect.bottom > window.innerHeight * 0.28;
  if (!visible) return;
  autoIndustry = (autoIndustry + 1) % industryCards.length;
  activateIndustry(autoIndustry);
}, 3200);

window.addEventListener("scroll", requestScrollUpdate, { passive: true });
window.addEventListener("resize", requestScrollUpdate);
requestScrollUpdate();
