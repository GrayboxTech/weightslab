import subprocess
import os


# --- Git Information Retrieval ---
def get_git_info():
    try:
        # Find git repository root by traversing up from current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        git_root = current_dir

        # Traverse up to find .git directory
        for _ in range(10): # Limit search depth
            if os.path.isdir(os.path.join(git_root, '.git')):
                break
            parent = os.path.dirname(git_root)
            if parent == git_root: # Reached filesystem root
                git_root = None
                break
            git_root = parent

        if git_root is None:
            return None, None, None

        # Get current git branch
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=git_root, stderr=subprocess.DEVNULL).strip().decode('utf-8')

        # Get current git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=git_root, stderr=subprocess.DEVNULL).strip().decode('utf-8')

        # Get version (you can modify this if you want a different versioning scheme)
        version = subprocess.check_output(['git', 'describe', '--tags', '--always'], cwd=git_root, stderr=subprocess.DEVNULL).strip().decode('utf-8')

        return branch, version, commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None, None


# --- Banner Definition ---
branch, version, commit_hash = get_git_info()

# ANSI styling. Color codes are zero-width, so alignment is always computed from
# the plain (uncolored) text length — never from the colored string.
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_RESET = "\x1b[0m"

# Rounded box-drawing characters (same look as the Claude Code banner).
_TL, _TR, _BL, _BR, _H, _V = "╭", "╮", "╰", "╯", "─", "│"

_SLOGAN = "Inspect - Edit - Evolve Neural Networks"
_CREDIT = "By GrayBx"

# "Weightslab" in the FIGlet "big_money-nw" font, shown as the wordmark INSIDE
# the box, above the slogan. Kept as a RAW string so the many backslashes in the
# glyphs stay literal (no invalid-escape warnings, no accidental line
# continuation). Do NOT reflow or re-indent — the columns must stay exactly as
# generated or the wordmark falls out of alignment.
_WORDMARK = r"""
$$\      $$\           $$\           $$\         $$\               $$\                 $$\
$$ | $\  $$ |          \__|          $$ |        $$ |              $$ |                $$ |
$$ |$$$\ $$ | $$$$$$\  $$\  $$$$$$\  $$$$$$$\  $$$$$$\    $$$$$$$\ $$ |       $$$$$$\  $$$$$$$\
$$ $$ $$\$$ |$$  __$$\ $$ |$$  __$$\ $$  __$$\ \_$$  _|  $$  _____|$$ |       \____$$\ $$  __$$\
$$$$  _$$$$ |$$$$$$$$ |$$ |$$ /  $$ |$$ |  $$ |  $$ |    \$$$$$$\  $$ |       $$$$$$$ |$$ |  $$ |
$$$  / \$$$ |$$   ____|$$ |$$ |  $$ |$$ |  $$ |  $$ |$$\  \____$$\ $$ |      $$  __$$ |$$ |  $$ |
$$  /   \$$ |\$$$$$$$\ $$ |\$$$$$$$ |$$ |  $$ |  \$$$$  |$$$$$$$  |$$$$$$$$\ \$$$$$$$ |$$$$$$$  |
\__/     \__| \_______|\__| \____$$ |\__|  \__|   \____/ \_______/ \________| \_______|\_______/
                           $$\   $$ |
                           \$$$$$$  |
                            \______/
"""
_wordmark_lines = _WORDMARK.strip("\n").split("\n")

# Fixed column spans of individual glyphs in the big_money-nw "WeightsLab"
# (letters laid out full-width so the spans don't bleed at a smush boundary):
# only the leading "W" is tinted green and the "L" red; everything else default.
_W_GLYPH_END = 13      # "W" occupies columns [0, 13)
_L_GLYPH_START = 67    # "L" occupies columns [67, 77)
_L_GLYPH_END = 77


def _colorize_wordmark_line(line: str) -> str:
    """Return `line` with only the 'W' glyph green and the 'L' glyph red."""
    return (
        f"{_GREEN}{line[:_W_GLYPH_END]}{_RESET}"
        f"{line[_W_GLYPH_END:_L_GLYPH_START]}"
        f"{_RED}{line[_L_GLYPH_START:_L_GLYPH_END]}{_RESET}"
        f"{line[_L_GLYPH_END:]}"
    )


def _package_version() -> str:
    """Best-effort current package version for the banner.

    Read from installed package metadata first, then the setuptools_scm-written
    ``_version`` module, then the git-describe fallback. Deliberately avoids
    ``import weightslab`` because this module is imported *during* the package's
    own initialization (before ``__version__`` is assigned there)."""
    try:
        from importlib.metadata import version as _dist_version, PackageNotFoundError
        try:
            return _dist_version("weightslab")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        from weightslab._version import __version__ as _scm_version  # written at build time
        return str(_scm_version)
    except Exception:
        pass
    return version or "dev"  # git describe (from get_git_info above)


# Credit shown in the BOTTOM border, with the version appended: "By GrayBx, vX.Y.Z".
_CREDIT_LABEL = f"{_CREDIT}, v{_package_version()}"

# Indent for the wordmark inside the box.
_PAD = "  "

# Inner width = visible columns between the two vertical borders. Sized to fit
# the widest of: the wordmark, the slogan, and the credit border. Alignment is
# always computed from PLAIN (uncolored) text lengths.
_INNER = max(
    [len(_PAD + l) for l in _wordmark_lines]
    + [len(_SLOGAN), len(_CREDIT_LABEL) + 5]
) + 2


def _plain_border(corner_l: str, corner_r: str) -> str:
    """A borderline with no embedded label: ``<corner>────────<corner>``."""
    return f"{corner_l}{_H * _INNER}{corner_r}"


def _titled_border(corner_l: str, corner_r: str, plain: str, colored: str) -> str:
    """A border with an embedded label: ``<corner>─── label ───<corner>``."""
    used = 3 + 1 + len(plain) + 1  # "───" + space + label + space
    fill = max(_INNER - used, 0)
    return f"{corner_l}{_H * 3} {colored} {_H * fill}{corner_r}"


def _row(plain: str = "", colored: str = None) -> str:
    """A content row padded to the inner width between the vertical borders."""
    if colored is None:
        colored = plain
    pad = max(_INNER - len(plain), 0)
    return f"{_V}{colored}{' ' * pad}{_V}"


_banner_lines = [""]
_banner_lines.append(_plain_border(_TL, _TR))  # top border: no title text
_banner_lines.append(_row())
# The "WeightsLab" wordmark, inside the box, above the slogan.
_banner_lines += [
    _row(_PAD + line, _PAD + _colorize_wordmark_line(line)) for line in _wordmark_lines
]
_banner_lines.append(_row())
_banner_lines.append(_row(_SLOGAN.center(_INNER)))  # slogan: centered, default color
_banner_lines.append(_row())
# "By GrayBx, vX.Y.Z" in the bottom border.
_banner_lines.append(_titled_border(_BL, _BR, _CREDIT_LABEL, _CREDIT_LABEL))
_banner_lines.append("")
_BANNER = "\n".join(_banner_lines) + "\n"
_BANNER__ = _BANNER # Expose banner with a different name for external use and legacy
