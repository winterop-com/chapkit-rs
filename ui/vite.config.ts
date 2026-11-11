import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { appearanceScript } from '@winterop-com/sokkel-ui/lib/appearance'
import { defineConfig } from 'vite'

/** The model service the dev server sends API calls to. */
const service = process.env.CHAPKIT_SERVICE ?? 'http://127.0.0.1:8766'

/** Wears the kept look on <html> before the page draws, so it never flashes the wrong scheme. */
const appearance = {
    name: 'appearance',
    transformIndexHtml: () => [{ tag: 'script', children: appearanceScript(), injectTo: 'head-prepend' as const }],
}

/** Served by the model binary at /ui/; built into the crate that embeds it. */
export default defineConfig({
    base: '/ui/',
    plugins: [react(), tailwindcss(), appearance],
    build: {
        outDir: '../crates/chapkit-ui/dist',
        emptyOutDir: true,
        chunkSizeWarningLimit: 1500,
    },
    server: {
        proxy: Object.fromEntries(['/api', '/health', '/metrics', '/docs', '/openapi.json'].map((path) => [path, service])),
    },
})
