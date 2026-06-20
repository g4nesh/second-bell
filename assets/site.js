const root = document.documentElement;
const progress = document.querySelector(".progress");
const reveals = document.querySelectorAll(".reveal");
const countEls = document.querySelectorAll("[data-count]");
const tabs = document.querySelectorAll(".clock-tab");
const panels = document.querySelectorAll(".clock-panel");

let latestScroll = 0;
let ticking = false;
const counted = new WeakSet();

function updateScrollState() {
  const max = Math.max(1, document.body.scrollHeight - window.innerHeight);
  const ratio = Math.min(1, Math.max(0, latestScroll / max));
  root.style.setProperty("--progress", ratio.toFixed(4));
  root.style.setProperty("--scroll", ratio.toFixed(4));
  if (progress) {
    progress.style.width = `${ratio * 100}%`;
  }
  ticking = false;
}

function requestScrollUpdate() {
  latestScroll = window.scrollY || window.pageYOffset;
  if (!ticking) {
    window.requestAnimationFrame(updateScrollState);
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
  { threshold: 0.16, rootMargin: "0px 0px -8% 0px" }
);

reveals.forEach((el) => revealObserver.observe(el));

function animateCount(el) {
  if (counted.has(el)) return;
  counted.add(el);
  const target = Number(el.dataset.count || 0);
  const start = performance.now();
  const duration = 1200;

  function frame(now) {
    const progressRatio = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - progressRatio, 3);
    el.textContent = String(Math.round(target * eased));
    if (progressRatio < 1) {
      window.requestAnimationFrame(frame);
    }
  }

  window.requestAnimationFrame(frame);
}

const countObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCount(entry.target);
      }
    });
  },
  { threshold: 0.75 }
);

countEls.forEach((el) => countObserver.observe(el));

function activateStep(index) {
  tabs.forEach((tab, i) => {
    const selected = i === index;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", selected ? "true" : "false");
  });
  panels.forEach((panel, i) => {
    panel.classList.toggle("active", i === index);
  });
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateStep(index));
});

let autoStep = 0;
window.setInterval(() => {
  const demo = document.querySelector(".clock-demo");
  if (!demo) return;
  const rect = demo.getBoundingClientRect();
  const visible = rect.top < window.innerHeight * 0.8 && rect.bottom > window.innerHeight * 0.2;
  if (!visible || document.hidden) return;
  autoStep = (autoStep + 1) % panels.length;
  activateStep(autoStep);
}, 4200);

window.addEventListener("scroll", requestScrollUpdate, { passive: true });
window.addEventListener("resize", requestScrollUpdate);
requestScrollUpdate();
