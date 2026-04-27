(function () {
  "use strict";

  window.showToast = function (message, type) {
    var container = document.getElementById("toast-container");
    if (!container) return;

    var toast = document.createElement("div");
    toast.className = "toast " + (type || "info");
    toast.textContent = message;
    container.appendChild(toast);

    window.setTimeout(function () {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(8px)";
      toast.style.transition = "all 0.25s ease";
      window.setTimeout(function () {
        toast.remove();
      }, 260);
    }, 3200);
  };

  function isEmailLike(value) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(value || "").trim());
  }

  function bindEmailCodeCooldown() {
    var form = document.getElementById("send-code-form");
    var emailInput = document.getElementById("register-email") || document.getElementById("register-email-send");
    var sendButton = document.getElementById("send-code-btn");
    if (!form || !emailInput || !sendButton) return;

    var cooldownSeconds = Number(form.getAttribute("data-cooldown-sec") || "60");
    var timer = null;

    function keyForEmail(email) {
      return "auth_email_code_cooldown:" + String(email || "").toLowerCase();
    }

    function currentEmail() {
      return String(emailInput.value || "").trim().toLowerCase();
    }

    function safeStorageGet(key) {
      try {
        return window.localStorage.getItem(key);
      } catch (_) {
        return null;
      }
    }

    function safeStorageSet(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (_) {
        // Ignore storage errors and keep server-side cooldown checks.
      }
    }

    function safeStorageRemove(key) {
      try {
        window.localStorage.removeItem(key);
      } catch (_) {
        // Ignore storage errors.
      }
    }

    function getRemainingSeconds(email) {
      if (!email) return 0;
      var raw = safeStorageGet(keyForEmail(email));
      if (!raw) return 0;
      var expireAt = Number(raw);
      if (!expireAt || Number.isNaN(expireAt)) return 0;
      var diff = Math.ceil((expireAt - Date.now()) / 1000);
      if (diff <= 0) {
        safeStorageRemove(keyForEmail(email));
        return 0;
      }
      return diff;
    }

    function setCooldown(email, seconds) {
      if (!email || seconds <= 0) return;
      var expireAt = Date.now() + seconds * 1000;
      safeStorageSet(keyForEmail(email), String(expireAt));
    }

    function renderButton() {
      var sec = getRemainingSeconds(currentEmail());
      if (sec > 0) {
        sendButton.disabled = true;
        sendButton.textContent = "\u91cd\u65b0\u53d1\u9001(" + sec + "s)";
      } else {
        sendButton.disabled = false;
        sendButton.textContent = "\u53d1\u9001\u9a8c\u8bc1\u7801";
      }
    }

    form.addEventListener("submit", function (evt) {
      var submitter = evt.submitter || document.activeElement;
      if (!submitter || submitter.id !== "send-code-btn") return;

      var emailNow = currentEmail();
      if (!emailNow) {
        evt.preventDefault();
        window.showToast("\u8bf7\u5148\u8f93\u5165\u90ae\u7bb1", "info");
        return;
      }
      if (!isEmailLike(emailNow)) {
        evt.preventDefault();
        window.showToast("\u90ae\u7bb1\u683c\u5f0f\u4e0d\u6b63\u786e", "info");
        return;
      }

      var sec = getRemainingSeconds(emailNow);
      if (sec > 0) {
        evt.preventDefault();
        window.showToast("\u8bf7 " + sec + " \u79d2\u540e\u518d\u53d1\u9001\u9a8c\u8bc1\u7801", "info");
      }
    });

    emailInput.addEventListener("input", renderButton);

    var successFlag = document.getElementById("email-code-sent-flag");
    if (successFlag) {
      var sentEmail = String(successFlag.getAttribute("data-email") || "").trim().toLowerCase();
      var sentCooldown = Number(successFlag.getAttribute("data-cooldown-sec") || String(cooldownSeconds));
      setCooldown(sentEmail, sentCooldown);
    }

    renderButton();
    timer = window.setInterval(renderButton, 1000);
    window.addEventListener("beforeunload", function () {
      if (timer) window.clearInterval(timer);
    });
  }

  function bindRegisterAvailabilityCheck() {
    var form = document.getElementById("send-code-form");
    var usernameInput = document.getElementById("register-username");
    var agentInput = document.getElementById("register-agent");
    var usernameHint = document.getElementById("register-username-hint");
    var agentHint = document.getElementById("register-agent-hint");
    if (!form || !usernameInput || !agentInput || !usernameHint || !agentHint) return;

    var timers = { username: null, agent_name: null };
    var requestSeq = { username: 0, agent_name: 0 };
    var cache = Object.create(null);
    var bypassSubmitGuard = false;

    function normalize(value) {
      return String(value || "").trim();
    }

    function setHint(hintEl, state, message) {
      hintEl.classList.remove("checking", "ok", "bad");
      if (state) {
        hintEl.classList.add(state);
      }
      hintEl.textContent = message;
    }

    function setInputState(inputEl, state) {
      inputEl.classList.remove("field-valid", "field-invalid");
      if (state === "ok") {
        inputEl.classList.add("field-valid");
      } else if (state === "bad") {
        inputEl.classList.add("field-invalid");
      }
    }

    function defaultHint(field) {
      if (field === "username") {
        return "\u81f3\u5c11 2 \u4e2a\u5b57\u7b26\uff0c\u8f93\u5165\u540e\u81ea\u52a8\u68c0\u67e5\u662f\u5426\u53ef\u7528";
      }
      return "\u5efa\u8bae\u4f7f\u7528\u6709\u8fa8\u8bc6\u5ea6\u7684 Agent \u540d\u79f0";
    }

    function checkAvailability(field, inputEl, hintEl, minLength) {
      var value = normalize(inputEl.value);
      var lowered = value.toLowerCase();
      if (!value) {
        setHint(hintEl, "", defaultHint(field));
        setInputState(inputEl, "");
        return Promise.resolve(true);
      }
      if (value.length < minLength) {
        setHint(hintEl, "bad", "\u81f3\u5c11 " + minLength + " \u4e2a\u5b57\u7b26");
        setInputState(inputEl, "bad");
        return Promise.resolve(false);
      }
      if (value.length > 50) {
        setHint(hintEl, "bad", "\u957f\u5ea6\u4e0d\u80fd\u8d85\u8fc7 50 \u4e2a\u5b57\u7b26");
        setInputState(inputEl, "bad");
        return Promise.resolve(false);
      }

      var cacheKey = field + "::" + lowered;
      if (cache[cacheKey]) {
        setHint(hintEl, cache[cacheKey].available ? "ok" : "bad", cache[cacheKey].message);
        setInputState(inputEl, cache[cacheKey].available ? "ok" : "bad");
        return Promise.resolve(cache[cacheKey].available);
      }

      requestSeq[field] += 1;
      var currentSeq = requestSeq[field];
      setHint(hintEl, "checking", "\u68c0\u67e5\u4e2d...");

      return fetch(
        "/register/check-availability?field=" + encodeURIComponent(field) + "&value=" + encodeURIComponent(value),
        { headers: { Accept: "application/json" } }
      )
        .then(function (res) {
          if (!res.ok) throw new Error("request failed");
          return res.json();
        })
        .then(function (data) {
          if (currentSeq !== requestSeq[field]) return false;

          var available = !!(data && data.ok && data.available);
          var message = data && data.message ? String(data.message) : available ? "\u53ef\u7528" : "\u4e0d\u53ef\u7528";
          cache[cacheKey] = { available: available, message: message };
          setHint(hintEl, available ? "ok" : "bad", message);
          setInputState(inputEl, available ? "ok" : "bad");
          return available;
        })
        .catch(function () {
          if (currentSeq !== requestSeq[field]) return false;
          setHint(hintEl, "bad", "\u68c0\u67e5\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5");
          setInputState(inputEl, "bad");
          return false;
        });
    }

    function scheduleCheck(field, inputEl, hintEl, minLength) {
      if (timers[field]) {
        window.clearTimeout(timers[field]);
      }
      timers[field] = window.setTimeout(function () {
        checkAvailability(field, inputEl, hintEl, minLength);
      }, 260);
    }

    usernameInput.addEventListener("input", function () {
      scheduleCheck("username", usernameInput, usernameHint, 2);
    });
    agentInput.addEventListener("input", function () {
      scheduleCheck("agent_name", agentInput, agentHint, 1);
    });
    usernameInput.addEventListener("blur", function () {
      checkAvailability("username", usernameInput, usernameHint, 2);
    });
    agentInput.addEventListener("blur", function () {
      checkAvailability("agent_name", agentInput, agentHint, 1);
    });

    form.addEventListener("submit", function (evt) {
      var submitter = evt.submitter || document.activeElement;
      if (submitter && submitter.id === "send-code-btn") return;

      if (bypassSubmitGuard) {
        bypassSubmitGuard = false;
        return;
      }

      evt.preventDefault();
      Promise.all([
        checkAvailability("username", usernameInput, usernameHint, 2),
        checkAvailability("agent_name", agentInput, agentHint, 1),
      ]).then(function (checks) {
        if (!checks[0] || !checks[1]) {
          window.showToast("\u8bf7\u4fee\u6b63\u7528\u6237\u540d\u6216 Agent \u540d\u79f0\u540e\u518d\u63d0\u4ea4", "error");
          return;
        }

        bypassSubmitGuard = true;
        if (typeof form.requestSubmit === "function" && submitter) {
          form.requestSubmit(submitter);
          return;
        }
        form.submit();
      });
    });
  }

  function bindHeroTypewriter() {
    var el = document.getElementById("hero-typewriter");
    if (!el) return;

    var raw = el.getAttribute("data-phrases") || "[]";
    var phrases;
    try {
      phrases = JSON.parse(raw);
    } catch (_) {
      phrases = [];
    }

    phrases = (phrases || []).filter(function (x) {
      return typeof x === "string" && x.trim().length > 0;
    });
    if (!phrases.length) return;

    var phraseIndex = 0;
    var charIndex = 0;
    var deleting = false;

    function tick() {
      var phrase = phrases[phraseIndex];
      var nextText;

      if (!deleting) {
        charIndex += 1;
        nextText = phrase.slice(0, charIndex);
        el.textContent = nextText;

        if (charIndex >= phrase.length) {
          deleting = true;
          window.setTimeout(tick, 1100);
          return;
        }

        window.setTimeout(tick, 95);
        return;
      }

      charIndex -= 1;
      nextText = phrase.slice(0, Math.max(charIndex, 0));
      el.textContent = nextText;

      if (charIndex <= 0) {
        deleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        window.setTimeout(tick, 280);
        return;
      }

      window.setTimeout(tick, 55);
    }

    el.textContent = "";
    tick();
  }

  function init() {
    bindEmailCodeCooldown();
    bindRegisterAvailabilityCheck();
    bindHeroTypewriter();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
