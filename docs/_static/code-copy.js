function detectLineKind(text) {
  const trimmed = text.replace(/\u00a0/g, " ").trimStart();
  if (trimmed.startsWith("+")) return "wl-line-added";
  if (trimmed.startsWith("-")) return "wl-line-removed";
  return "";
}

function isMarkedDiffBlock(block) {
  return Boolean(block.closest(".wl-diff-lines"));
}

function decoratePrefixedDiffRows(block) {
  if (!isMarkedDiffBlock(block)) return;
  const pre = block.querySelector("pre");
  if (!pre || pre.dataset.wlDiffDecorated === "1") return;

  const nodes = Array.from(pre.childNodes);
  const rows = [];
  let current = null;

  for (const node of nodes) {
    if (node.nodeType === Node.ELEMENT_NODE && node.classList.contains("linenos")) {
      current = { lineno: node, nodes: [] };
      rows.push(current);
      continue;
    }
    if (current) current.nodes.push(node);
  }

  for (const row of rows) {
    if (row.nodes.length === 0) continue;
    const kind = detectLineKind(row.nodes.map((n) => n.textContent || "").join(""));
    if (!kind) continue;

    row.lineno.classList.add(kind);

    for (const node of row.nodes) {
      if (node.nodeType === Node.ELEMENT_NODE) {
        node.classList.add("wl-line-segment", kind);
      } else if (node.nodeType === Node.TEXT_NODE && node.textContent) {
        const segment = document.createElement("span");
        segment.className = `wl-line-segment ${kind}`;
        segment.textContent = node.textContent;
        pre.replaceChild(segment, node);
      }
    }
  }

  pre.dataset.wlDiffDecorated = "1";
}

function isDiffLikeBlock(block) {
  return (
    isMarkedDiffBlock(block) ||
    block.classList.contains("highlight-diff") ||
    block.querySelector(".gd, .gi, .wl-line-added, .wl-line-removed")
  );
}

document.addEventListener("DOMContentLoaded", () => {
  const blocks = document.querySelectorAll("div.highlight");
  for (const block of blocks) {
    decoratePrefixedDiffRows(block);
    if (block.querySelector(".wl-copy-btn")) continue;

    const pre = block.querySelector("pre");
    if (!pre) continue;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "wl-copy-btn";
    button.textContent = "Copy";

    button.addEventListener("click", async () => {
      try {
        const lines = pre.querySelectorAll(".linenos");
        const previous = [];
        for (const line of lines) {
          previous.push(line.textContent);
          line.textContent = "";
        }

        let text = pre.innerText.replace(/\u00a0/g, " ");
        if (isDiffLikeBlock(block)) {
          const cleanedLines = text
            .split(/\r?\n/)
            .filter((line) => {
              const trimmed = line.trimStart();
              return !(trimmed.startsWith("-") && !trimmed.startsWith("--"));
            })
            .map((line) => line.replace(/^(\s*)\+\s{0,2}/, "$1"));

          const compacted = [];
          for (const line of cleanedLines) {
            const isBlank = line.trim() === "";
            if (isBlank && compacted.length > 0 && compacted[compacted.length - 1].trim() === "") {
              continue;
            }
            compacted.push(line);
          }
          text = compacted.join("\n");
        }
        text = text.trimEnd();

        lines.forEach((line, i) => {
          line.textContent = previous[i];
        });

        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
        setTimeout(() => {
          button.textContent = "Copy";
        }, 1200);
      } catch (_err) {
        button.textContent = "Failed";
        setTimeout(() => {
          button.textContent = "Copy";
        }, 1200);
      }
    });

    block.appendChild(button);
  }
});
