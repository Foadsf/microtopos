// MicroTopos website: theme switch, mobile menu, screenshot lightbox, copy buttons, outline highlight.
(() => {
  const root = document.documentElement;
  const store = { get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
                  set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } } };

  const theme = document.querySelector(".theme-toggle");
  if (theme) theme.addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    store.set("mt-theme", root.dataset.theme);
  });

  const toggle = document.querySelector(".nav-toggle");
  const nav = document.getElementById("site-nav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { nav.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); } });
  }

  // wide tables scroll inside their own box on phones instead of widening the page
  document.querySelectorAll("table.tableblock").forEach((t) => {
    const w = document.createElement("div");
    w.className = "tableblock-wrap";
    t.parentNode.insertBefore(w, t);
    w.appendChild(t);
  });

  const box = document.querySelector(".lightbox");
  if (box) {
    const img = box.querySelector("img"), cap = box.querySelector("p");
    document.querySelectorAll(".imageblock img").forEach((el) => {
      el.addEventListener("click", () => {
        img.src = el.currentSrc || el.src;
        img.alt = el.alt;
        const t = el.closest(".imageblock").querySelector(".title");
        cap.textContent = t ? t.textContent : el.alt;
        box.hidden = false;
      });
    });
    const close = () => { box.hidden = true; img.removeAttribute("src"); };
    box.addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !box.hidden) close(); });
  }

  document.querySelectorAll(".listingblock").forEach((b) => {
    const pre = b.querySelector("pre");
    if (!pre || !navigator.clipboard) return;
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(pre.innerText); btn.textContent = "Copied"; }
      catch (e) { btn.textContent = "Select and copy"; }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
    b.appendChild(btn);
  });

  const links = [...document.querySelectorAll(".outline a")];
  if (links.length && "IntersectionObserver" in window) {
    const byId = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const seen = new Set();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? seen.add(e.target.id) : seen.delete(e.target.id)));
      const first = [...byId.keys()].find((id) => seen.has(id));
      if (first) links.forEach((a) => a.classList.toggle("active", a === byId.get(first)));
    }, { rootMargin: "-70px 0px -65% 0px" });
    byId.forEach((_, id) => { const h = document.getElementById(id); if (h) io.observe(h); });
  }
})();
