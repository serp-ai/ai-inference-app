// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  modules: [
    '@nuxt/ui',
    '@nuxt/eslint',
    '@compodium/nuxt',
  ],
  devtools: { enabled: true },
  css: ['~/assets/main.css'],

  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000',
    },
  },

  ssr: false, // Disable SSR for SPA mode

  future: { compatibilityVersion: 4 },
  compatibilityDate: '2025-07-15',

  // Development config
  eslint: {
    config: {
      stylistic: {
        quotes: 'single',
        commaDangle: 'never',
      },
    },
  },
})
