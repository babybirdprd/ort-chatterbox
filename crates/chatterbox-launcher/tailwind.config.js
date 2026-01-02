/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./src/**/*.{rs,html,css}", "./assets/**/*.html"],
    theme: {
        extend: {
            colors: {
                // Modern dark theme
                'surface': {
                    50: '#f8fafc',
                    100: '#f1f5f9',
                    200: '#e2e8f0',
                    800: '#1e293b',
                    900: '#0f172a',
                    950: '#020617',
                },
                'accent': {
                    400: '#a78bfa',
                    500: '#8b5cf6',
                    600: '#7c3aed',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'gradient': 'gradient 8s ease infinite',
            },
            keyframes: {
                gradient: {
                    '0%, 100%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                },
            },
            backgroundSize: {
                '300%': '300%',
            },
        },
    },
    plugins: [],
}
