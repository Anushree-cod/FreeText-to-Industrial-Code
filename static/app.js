(() => {
  const form = document.getElementById("search-form");
  const descriptionEl = document.getElementById("description");
  const languageEl = document.getElementById("language");
  const codeSystemEl = document.getElementById("code-system");
  const submitLabel = document.getElementById("submit-label");
  const spinner = document.getElementById("spinner");
  const errorBanner = document.getElementById("error-banner");
  const infoBanner = document.getElementById("info-banner");

  const noResultsEl = document.getElementById("no-results");
  const resultsEl = document.getElementById("results");
  const primaryCodeEl = document.getElementById("primary-code");
  const confidenceLabelEl = document.getElementById("confidence-label");
  const confidenceBarEl = document.getElementById("confidence-bar");
  const similarityEl = document.getElementById("similarity");
  const detectedLanguageEl = document.getElementById("detected-language");
  const rationaleEl = document.getElementById("rationale");
  const suggestionsEl = document.getElementById("suggestions");
  const neighborsEl = document.getElementById("neighbors");

  const feedbackButtons = document.querySelectorAll(".feedback-buttons .btn");
  const userCodeEl = document.getElementById("user-code");
  const userCommentEl = document.getElementById("user-comment");
  const feedbackStatusEl = document.getElementById("feedback-status");

  const siteNav = document.getElementById("site-nav");
  const landing = document.getElementById("landing");
  const heroTitle = document.getElementById("hero-title");
  const heroChrome = document.getElementById("hero-chrome");
  const heroSides = document.getElementById("hero-sides");
  const heroTagline = document.getElementById("hero-tagline");
  const skyline = document.getElementById("skyline");
  const scrollCue = document.getElementById("scroll-cue");

  let lastResultPayload = null;
  let isSubmitting = false;
  const recentSearches = [];

  /* ---------- Sticky landing zoom (21hrs-style) ---------- */
  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  function onScroll() {
    if (!landing) return;

    const rect = landing.getBoundingClientRect();
    const total = landing.offsetHeight - window.innerHeight;
    const raw = total > 0 ? -rect.top / total : 0;
    const p = easeInOut(clamp(raw, 0, 1));

    // Skyline zooms in hard — "land on the factories"
    if (skyline) {
      const scale = 1 + p * 1.55;
      const y = p * 18;
      skyline.style.transform = `translateX(-50%) translateY(${y}vh) scale(${scale})`;
    }

    // Title rises and fades as we dive past it into the skyline
    if (heroTitle) {
      heroTitle.style.transform = `translateY(${p * -28}vh) scale(${1 + p * 0.35})`;
      heroTitle.style.opacity = String(1 - p * 1.15);
    }

    const chromeOpacity = String(Math.max(0, 1 - p * 1.4));
    if (heroChrome) heroChrome.style.opacity = chromeOpacity;
    if (heroSides) heroSides.style.opacity = chromeOpacity;
    if (heroTagline) heroTagline.style.opacity = chromeOpacity;

    if (scrollCue) {
      scrollCue.classList.toggle("fade", raw > 0.08);
    }

    if (siteNav) {
      siteNav.classList.toggle("visible", raw > 0.55 || window.scrollY > window.offsetHeight * 0.55);
    }
  }

  let ticking = false;
  window.addEventListener(
    "scroll",
    () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        onScroll();
        ticking = false;
      });
    },
    { passive: true }
  );
  onScroll();

  if (scrollCue) {
    scrollCue.addEventListener("click", () => {
      const classify = document.getElementById("classify");
      if (!classify || !landing) return;
      // Jump past the landing scroll track into classify
      const target = landing.offsetTop + landing.offsetHeight - window.innerHeight * 0.05;
      window.scrollTo({ top: target, behavior: "smooth" });
    });
  }

  const sections = document.querySelectorAll(".section");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("in-view");
      });
    },
    { threshold: 0.15 }
  );
  sections.forEach((s) => observer.observe(s));

  /* ---------- Classify UI ---------- */
  function setLoading(loading) {
    isSubmitting = loading;
    if (submitLabel) submitLabel.textContent = loading ? "Running…" : "Get industrial code";
    if (spinner) spinner.classList.toggle("hidden", !loading);
  }

  function showError(message) {
    if (errorBanner) {
      errorBanner.textContent = message;
      errorBanner.classList.remove("hidden");
    }
    infoBanner?.classList.add("hidden");
  }

  function showInfo(message) {
    if (infoBanner) {
      infoBanner.textContent = message;
      infoBanner.classList.remove("hidden");
    }
    errorBanner?.classList.add("hidden");
  }

  function clearBanners() {
    errorBanner?.classList.add("hidden");
    infoBanner?.classList.add("hidden");
  }

  function renderResults(payload) {
    lastResultPayload = payload;
    if (!payload) return;

    noResultsEl?.classList.add("hidden");
    resultsEl?.classList.remove("hidden");

    if (primaryCodeEl) {
      primaryCodeEl.textContent = `${payload.top_code} — ${payload.top_title}`;
    }
    if (confidenceLabelEl) confidenceLabelEl.textContent = `${payload.confidence}%`;
    if (confidenceBarEl) {
      confidenceBarEl.style.width = `${payload.confidence}%`;
      confidenceBarEl.classList.toggle("low", payload.confidence < 35);
    }
    if (similarityEl) similarityEl.textContent = payload.similarity.toFixed(4);
    if (detectedLanguageEl) {
      detectedLanguageEl.textContent = payload.detected_language || "unknown";
    }
    if (rationaleEl) rationaleEl.textContent = payload.rationale || "";

    if (suggestionsEl) {
      suggestionsEl.innerHTML = "";
      (payload.top_suggestions || []).forEach((s, idx) => {
        const li = document.createElement("li");
        const label = s.source ? `(${s.source})` : "";
        li.innerHTML = `<div class="suggestion-main">
  <span class="rank">${idx + 1}.</span>
  <span class="code">${s.code}</span>
  <span class="source-label">${label}</span>
  <span class="title">${s.title}</span>
</div>
<div class="suggestion-sub">
  <span class="short-desc">${s.short_description || ""}</span>
  <span class="sim-tag">sim=${s.similarity.toFixed(4)}</span>
</div>`;
        suggestionsEl.appendChild(li);
      });
    }

    if (neighborsEl) {
      neighborsEl.innerHTML = "";
      (payload.neighbors || []).forEach((n) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="code">${n.code}</span> — <span class="title">${n.title}</span> <span class="sim">(sim=${n.sim.toFixed(4)})</span>`;
        neighborsEl.appendChild(li);
      });
    }

    const sectorNewsBlock = document.getElementById("sector-news-block");
    const sectorNewsContent = document.getElementById("sector-news-content");
    const sectorNewsEmpty = document.getElementById("sector-news-empty");

    if (sectorNewsBlock && sectorNewsContent && sectorNewsEmpty) {
      if (payload.sector_news && payload.sector_news.length > 0) {
        sectorNewsBlock.hidden = false;
        sectorNewsEmpty.hidden = true;
        sectorNewsContent.innerHTML = "";
        payload.sector_news.forEach((news) => {
          const item = document.createElement("div");
          item.className = "news-item";
          item.innerHTML = `
            <h5 class="news-title"><a href="${news.link || "#"}" target="_blank" rel="noopener noreferrer">${news.title || "No title"}</a></h5>
            <span class="news-source">${news.source || ""}</span>
            <p class="news-snippet">${news.snippet || ""}</p>
            ${news.date ? `<span class="news-date">${news.date}</span>` : ""}
          `;
          sectorNewsContent.appendChild(item);
        });
      } else if (payload.sector_news == null) {
        sectorNewsBlock.hidden = true;
      } else {
        sectorNewsBlock.hidden = false;
        sectorNewsContent.innerHTML = "";
        sectorNewsEmpty.hidden = false;
      }
    }

    addToRecentSearches(descriptionEl ? descriptionEl.value : "", payload);
    const quick = document.getElementById("quick-actions-section");
    if (quick) quick.hidden = false;
  }

  function addToRecentSearches(query, result) {
    if (!query || !result) return;
    recentSearches.unshift({
      query: query.length > 60 ? query.slice(0, 60) + "..." : query,
      fullQuery: query,
      code: result.top_code,
      title: result.top_title,
      timestamp: new Date().toLocaleTimeString(),
    });
    if (recentSearches.length > 5) recentSearches.pop();
    updateRecentSearchesDisplay();
  }

  function updateRecentSearchesDisplay() {
    const listEl = document.getElementById("recent-searches-list");
    if (!listEl) return;
    if (!recentSearches.length) {
      listEl.innerHTML = '<p class="muted" id="no-recent-searches">No recent searches yet</p>';
      return;
    }
    listEl.innerHTML = "";
    recentSearches.forEach((search) => {
      const item = document.createElement("div");
      item.className = "recent-search-item";
      item.innerHTML = `
        <div class="recent-search-header">
          <span class="recent-search-code">${search.code}</span>
          <span class="recent-search-time">${search.timestamp}</span>
        </div>
        <div class="recent-search-query">${search.query}</div>
        <div class="recent-search-title">${search.title}</div>
      `;
      item.addEventListener("click", () => {
        if (descriptionEl) {
          descriptionEl.value = search.fullQuery;
          descriptionEl.focus();
        }
      });
      listEl.appendChild(item);
    });
  }

  const actionStatus = document.getElementById("action-status");

  document.getElementById("copy-code-btn")?.addEventListener("click", () => {
    if (!lastResultPayload) return;
    const text = `${lastResultPayload.top_code} - ${lastResultPayload.top_title}`;
    navigator.clipboard.writeText(text).then(() => {
      if (actionStatus) {
        actionStatus.textContent = "Copied!";
        setTimeout(() => { actionStatus.textContent = ""; }, 2000);
      }
    });
  });

  async function loadDatabaseStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const stats = await res.json();
      const map = {
        "total-codes": stats.total_codes,
        "nic-codes": stats.nic_codes,
        "naics-codes": stats.naics_codes,
      };
      Object.entries(map).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val ?? "-";
      });
    } catch {
      /* ignore */
    }
  }
  loadDatabaseStats();

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isSubmitting) return;

    const description = descriptionEl?.value.trim() || "";
    if (!description) {
      showError("Please enter a business or activity description.");
      return;
    }

    clearBanners();
    if (feedbackStatusEl) feedbackStatusEl.textContent = "";
    setLoading(true);

    try {
      const response = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: description,
          language: languageEl?.value || "auto",
          code_system: codeSystemEl?.value || "both",
        }),
      });
      if (!response.ok) throw new Error("Classification failed. Please try again.");
      const data = await response.json();
      renderResults(data);
      if (data.confidence < 35) {
        showInfo("Low confidence — add more detail about products, customers, and scale.");
      }
      resultsEl?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  });

  feedbackButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!lastResultPayload) {
        if (feedbackStatusEl) {
          feedbackStatusEl.textContent = "Run a classification first.";
        }
        return;
      }
      if (feedbackStatusEl) feedbackStatusEl.textContent = "Sending…";
      try {
        const res = await fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: descriptionEl?.value || "",
            rating: btn.dataset.rating,
            model_top_code: lastResultPayload.top_code,
            user_code: userCodeEl?.value || null,
            comment: userCommentEl?.value || null,
          }),
        });
        if (!res.ok) throw new Error();
        if (feedbackStatusEl) feedbackStatusEl.textContent = "Thanks for your feedback!";
        if (userCommentEl) userCommentEl.value = "";
      } catch {
        if (feedbackStatusEl) {
          feedbackStatusEl.textContent = "Could not send feedback.";
        }
      }
    });
  });

  document.querySelectorAll(".example").forEach((el) => {
    el.addEventListener("click", () => {
      const example = el.getAttribute("data-example");
      if (descriptionEl && example) {
        descriptionEl.value = example;
        descriptionEl.focus();
      }
    });
  });
})();
