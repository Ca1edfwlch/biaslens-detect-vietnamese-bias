import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Lighter dark theme -- was #080910..#1f2130 (near-black); raised
        // luminance while keeping the same relative depth steps between
        // levels, so nesting (panel-on-page, row-on-panel) still reads.
        bg: {
          0: '#14161f',
          1: '#191c26',
          2: '#20232f',
          3: '#282c3a',
          4: '#333749',
        },
        text: {
          1: 'rgba(255,255,255,0.92)',
          2: 'rgba(255,255,255,0.6)',
          // 3 and 4 were previously 0.35/0.18 -- measured contrast against
          // bg-2 was 3.22:1 and 1.73:1, both well under WCAG AA's 4.5:1 for
          // body text. Raised to hit ~4.5:1 and ~3.3:1ish (re-checked against
          // the lighter bg-2 above -- required alpha barely moved).
          3: 'rgba(255,255,255,0.46)',
          4: 'rgba(255,255,255,0.36)',
        },
        border: {
          // Previously 0.07/0.12 (contrast ~1.2:1 / ~1.4:1 against bg-2 --
          // essentially invisible as dividers). Raised to ~2.5:1 / ~3.8:1.
          DEFAULT: 'rgba(255,255,255,0.28)',
          subtle: 'rgba(255,255,255,0.40)',
        },
        accent: {
          DEFAULT: '#5b4fcf',
          hover: '#6a5dd8',
          dim: 'rgba(91,79,207,0.15)',
          border: 'rgba(91,79,207,0.3)',
          text: '#a89ef5',
        },
        status: {
          green: '#6fb96f',
          amber: '#e9a84a',
          red: '#e06060',
        },
      },
      fontFamily: {
        sans: ['var(--font-dm-sans)', 'sans-serif'],
        mono: ['var(--font-jetbrains-mono)', 'monospace'],
      },
      borderRadius: {
        sm: '8px',
        md: '11px',
        lg: '14px',
        xl: '18px',
      },
      keyframes: {
        pulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
        thinking: {
          '0%, 60%, 100%': { opacity: '0.3', transform: 'scale(0.8)' },
          '30%': { opacity: '1', transform: 'scale(1.1)' },
        },
      },
      animation: {
        pulse: 'pulse 2s infinite ease-in-out',
        thinking: 'thinking 1.2s infinite ease-in-out',
      },
    },
  },
  plugins: [],
}

export default config
