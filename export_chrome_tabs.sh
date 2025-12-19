#!/bin/bash
# Export all Chrome tabs from macOS
# Usage: ./export_chrome_tabs.sh > macos_tabs.txt

osascript <<EOF
tell application "Google Chrome"
    set tabList to ""
    repeat with w in windows
        repeat with t in tabs of w
            set tabList to tabList & (URL of t) & "\n"
        end repeat
    end repeat
    return tabList
end tell
EOF