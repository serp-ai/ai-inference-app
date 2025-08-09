<script setup lang="ts">
useHead({
  title: 'AI Text-to-Image',
  meta: [
    { name: 'description', content: 'Generate images from text using Qwen AI image generation' },
  ],
})

const toast = useToast()
const config = useRuntimeConfig()

// Basic settings (always visible)
const prompt = ref('')
const negativePrompt = ref('')
const steps = ref(20)
const cfg = ref(2.5)
const width = ref(1328)
const height = ref(1328)

const loading = ref(false)
const generatedImages = ref(null)
const generationHistory = ref([])

// Advanced settings (behind toggle)
const showAdvancedSettings = ref(false)
const seed = ref()
const samplerName = ref('euler')
const schedulerName = ref('simple')
const shift = ref(3.1)

// Download image function
function downloadImage(imageUrl, index = 0) {
  const link = document.createElement('a')
  link.href = imageUrl
  link.download = `qwen-generated-${Date.now()}-${index + 1}.png`
  link.click()
}

async function generateImage() {
  if (loading.value)
    return

  // Validate requirements
  if (!prompt.value.trim()) {
    toast.add({
      title: 'Prompt Required',
      description: 'Please enter a description of what you want to generate',
      color: 'red',
    })
    return
  }

  loading.value = true

  try {
    const requestBody = {
      prompt: prompt.value,
      negative_prompt: negativePrompt.value,
      steps: steps.value,
      cfg: cfg.value,
      width: width.value,
      height: height.value,
      seed: seed.value ? Number.parseInt(seed.value) : null,
      sampler_name: samplerName.value,
      scheduler_name: schedulerName.value,
      shift: shift.value,
    }

    const response = await $fetch(`${config.public.apiBase}/api/qwen-image/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: requestBody
    })

    // Convert base64 images to data URLs
    const imageUrls = response.images.map(b64 => `data:image/png;base64,${b64}`)
    generatedImages.value = imageUrls

    const newGeneration = {
      images: imageUrls,
      prompt: prompt.value,
      steps: steps.value,
      cfg: cfg.value,
      width: width.value,
      height: height.value,
      timestamp: new Date().toISOString(),
    }

    generationHistory.value.unshift(newGeneration)

    toast.add({
      title: 'Success!',
      description: `Generated ${imageUrls.length} image${imageUrls.length > 1 ? 's' : ''} successfully`,
      color: 'green',
    })
  }
  catch (error) {
    toast.add({
      title: 'Generation Failed',
      description: error.message || 'Something went wrong',
      color: 'red',
    })
  }
  finally {
    loading.value = false
  }
}

const qualityLabel = computed(() => {
  if (steps.value <= 10)
    return 'Fast'
  if (steps.value <= 20)
    return 'Balanced'
  return 'High Quality'
})

// Dimension presets
const dimensionPresets = [
  { name: 'Square', width: 1328, height: 1328 },
  { name: 'Portrait', width: 1024, height: 1536 },
  { name: 'Landscape', width: 1536, height: 1024 },
  { name: 'Widescreen', width: 1792, height: 1024 },
]

function setDimensionPreset(preset) {
  width.value = preset.width
  height.value = preset.height
}
</script>

<template>
  <div class="min-h-screen py-12">
    <div class="mx-auto max-w-6xl px-4">
      <!-- Header -->
      <div class="text-center mb-12">
        <div class="flex items-center justify-center mb-6">
          <div class="relative">
            <div
              class="w-16 h-16 rounded-full bg-gradient-to-br from-green-500/20 to-blue-500/20 flex items-center justify-center">
              <UIcon name="i-lucide-image" class="size-8 text-green-400" />
            </div>
            <div class="absolute -top-1 -right-1 animate-pulse">
              <div
                class="w-6 h-6 bg-gradient-to-r from-green-400 to-blue-400 rounded-full flex items-center justify-center">
                <UIcon name="i-lucide-sparkles" class="size-3 text-black animate-spin" />
              </div>
            </div>
          </div>
        </div>

        <h1
          class="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
          AI Text-to-Image
        </h1>

        <p class="text-lg text-muted max-w-2xl mx-auto mb-6">
          Generate stunning images from text descriptions using Qwen AI image generation
        </p>

        <div class="flex items-center justify-center gap-6 text-sm">
          <div class="flex items-center gap-2 text-green-400">
            <UIcon name="i-lucide-type" class="size-4" />
            <span>Text-to-Image</span>
          </div>
          <div class="flex items-center gap-2 text-blue-400">
            <UIcon name="i-lucide-image" class="size-4" />
            <span>High Resolution</span>
          </div>
          <div class="flex items-center gap-2 text-purple-400">
            <UIcon name="i-lucide-sliders" class="size-4" />
            <span>Fine Controls</span>
          </div>
        </div>
      </div>

      <div class="space-y-6">
        <!-- Input Form -->
        <SpotlightCard>
          <div class="flex items-center justify-between mb-6">
            <h3 class="text-xl font-semibold text-white flex items-center gap-2">
              <UIcon name="i-lucide-edit" class="size-5 text-green-400" />
              Create Your Image
            </h3>
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span class="text-xs text-green-400 font-medium">Ready</span>
            </div>
          </div>

          <form class="space-y-6" @submit.prevent="generateImage">
            <!-- Prompt Input -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Prompt
                <span class="text-red-400">*</span>
              </label>
              <UTextarea
                v-model="prompt"
                placeholder="A beautiful landscape with mountains and a sunset, highly detailed, photorealistic..."
                rows="3"
                class="w-full"
              />
              <p class="text-xs text-muted mt-1">
                Describe what you want to generate in detail
              </p>
            </div>

            <!-- Negative Prompt -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Negative Prompt (optional)
              </label>
              <UTextarea
                v-model="negativePrompt"
                placeholder="blurry, low quality, distorted, ugly..."
                rows="2"
                class="w-full"
              />
              <p class="text-xs text-muted mt-1">
                Describe what you want to avoid in the generation
              </p>
            </div>

            <!-- Steps Control -->
            <div>
              <div class="flex items-center justify-between mb-3">
                <label class="block text-sm font-medium text-white">Diffusion Steps</label>
                <div class="flex items-center gap-2">
                  <span class="text-sm text-muted">{{ steps }}</span>
                  <div class="px-2 py-1 bg-yellow-500/20 rounded text-xs text-yellow-400">
                    {{ qualityLabel }}
                  </div>
                </div>
              </div>
              <USlider v-model="steps" :min="1" :max="100" :step="1" class="w-full" />
              <div class="flex justify-between text-xs text-muted mt-1">
                <span>Faster</span>
                <span>Higher Quality</span>
              </div>
              <p class="text-xs text-muted mt-1">
                More steps generally produce higher quality images but take longer
              </p>
            </div>

            <!-- CFG Scale -->
            <div>
              <div class="flex items-center justify-between mb-3">
                <label class="block text-sm font-medium text-white">CFG Scale</label>
                <span class="text-sm text-muted">{{ cfg.toFixed(1) }}</span>
              </div>
              <USlider v-model="cfg" :min="1.0" :max="10.0" :step="0.1" class="w-full" />
              <div class="flex justify-between text-xs text-muted mt-1">
                <span>More Creative</span>
                <span>Follow Prompt</span>
              </div>
              <p class="text-xs text-muted mt-1">
                Higher values follow the prompt more closely, lower values are more creative
              </p>
            </div>

            <!-- Image Dimensions -->
            <div>
              <label class="block text-sm font-medium text-white mb-3">Image Dimensions</label>
              
              <!-- Preset Buttons -->
              <div class="flex flex-wrap gap-2 mb-3">
                <button
                  v-for="preset in dimensionPresets"
                  :key="preset.name"
                  type="button"
                  class="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                  :class="{ 'bg-green-600 hover:bg-green-700': width === preset.width && height === preset.height }"
                  @click="setDimensionPreset(preset)"
                >
                  {{ preset.name }} ({{ preset.width }}×{{ preset.height }})
                </button>
              </div>

              <!-- Custom Dimensions -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium text-white mb-2">Width</label>
                  <UInput v-model.number="width" type="number" :min="64" :max="2048" :step="64" class="w-full" />
                </div>
                <div>
                  <label class="block text-sm font-medium text-white mb-2">Height</label>
                  <UInput v-model.number="height" type="number" :min="64" :max="2048" :step="64" class="w-full" />
                </div>
              </div>
              <p class="text-xs text-muted mt-1">
                Larger dimensions produce more detailed images but take longer to generate
              </p>
            </div>

            <!-- Advanced Settings Toggle -->
            <div class="border-t border-gray-700 pt-4">
              <button type="button" @click="showAdvancedSettings = !showAdvancedSettings"
                class="flex items-center justify-between w-full p-3 bg-gray-800/50 hover:bg-gray-800 rounded-lg transition-colors">
                <div class="flex items-center gap-2">
                  <UIcon name="i-lucide-settings" class="size-4 text-gray-400" />
                  <span class="text-sm font-medium text-white">Advanced Settings</span>
                </div>
                <UIcon name="i-lucide-chevron-down" class="size-4 text-gray-400 transition-transform"
                  :class="{ 'rotate-180': showAdvancedSettings }" />
              </button>

              <p class="text-xs text-muted mt-2">
                Fine-tune sampling parameters and generation settings
              </p>
            </div>

            <!-- Advanced Settings Panel -->
            <div v-show="showAdvancedSettings" class="space-y-4 bg-gray-900/50 p-4 rounded-lg">
              <!-- Sampling Controls -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium text-white mb-2">Sampler</label>
                  <USelect 
                    v-model="samplerName" 
                    :options="[
                      { label: 'Euler', value: 'euler' },
                      { label: 'Euler CFG++', value: 'euler_cfg_pp' },
                      { label: 'Euler Ancestral', value: 'euler_ancestral' },
                      { label: 'Euler Ancestral CFG++', value: 'euler_ancestral_cfg_pp' },
                      { label: 'Heun', value: 'heun' },
                      { label: 'Heun++2', value: 'heunpp2' },
                      { label: 'DPM 2', value: 'dpm_2' },
                      { label: 'DPM 2 Ancestral', value: 'dpm_2_ancestral' },
                      { label: 'LMS', value: 'lms' },
                      { label: 'DPM Fast', value: 'dpm_fast' },
                      { label: 'DPM Adaptive', value: 'dpm_adaptive' },
                      { label: 'DPM++ 2S Ancestral', value: 'dpmpp_2s_ancestral' },
                      { label: 'DPM++ 2S Ancestral CFG++', value: 'dpmpp_2s_ancestral_cfg_pp' },
                      { label: 'DPM++ SDE', value: 'dpmpp_sde' },
                      { label: 'DPM++ SDE GPU', value: 'dpmpp_sde_gpu' },
                      { label: 'DPM++ 2M', value: 'dpmpp_2m' },
                      { label: 'DPM++ 2M CFG++', value: 'dpmpp_2m_cfg_pp' },
                      { label: 'DPM++ 2M SDE', value: 'dpmpp_2m_sde' },
                      { label: 'DPM++ 2M SDE GPU', value: 'dpmpp_2m_sde_gpu' },
                      { label: 'DPM++ 3M SDE', value: 'dpmpp_3m_sde' },
                      { label: 'DPM++ 3M SDE GPU', value: 'dpmpp_3m_sde_gpu' },
                      { label: 'DDPM', value: 'ddpm' },
                      { label: 'LCM', value: 'lcm' },
                      { label: 'iPNDM', value: 'ipndm' },
                      { label: 'iPNDM v', value: 'ipndm_v' },
                      { label: 'DEIS', value: 'deis' },
                      { label: 'Res Multistep', value: 'res_multistep' },
                      { label: 'Res Multistep CFG++', value: 'res_multistep_cfg_pp' },
                      { label: 'Res Multistep Ancestral', value: 'res_multistep_ancestral' },
                      { label: 'Res Multistep Ancestral CFG++', value: 'res_multistep_ancestral_cfg_pp' },
                      { label: 'Gradient Estimation', value: 'gradient_estimation' },
                      { label: 'Gradient Estimation CFG++', value: 'gradient_estimation_cfg_pp' },
                      { label: 'ER SDE', value: 'er_sde' },
                      { label: 'Seeds 2', value: 'seeds_2' },
                      { label: 'Seeds 3', value: 'seeds_3' },
                      { label: 'SA Solver', value: 'sa_solver' },
                      { label: 'SA Solver PECE', value: 'sa_solver_pece' },
                      { label: 'DDIM', value: 'ddim' },
                      { label: 'UniPC', value: 'uni_pc' },
                      { label: 'UniPC BH2', value: 'uni_pc_bh2' },
                    ]"
                    class="w-full" 
                  />
                  <p class="text-xs text-muted mt-1">
                    Sampling algorithm to use for generation
                  </p>
                </div>

                <div>
                  <label class="block text-sm font-medium text-white mb-2">Scheduler</label>
                  <USelect 
                    v-model="schedulerName" 
                    :options="[
                      { label: 'Simple', value: 'simple' },
                      { label: 'SGM Uniform', value: 'sgm_uniform' },
                      { label: 'Karras', value: 'karras' },
                      { label: 'Exponential', value: 'exponential' },
                      { label: 'DDIM Uniform', value: 'ddim_uniform' },
                      { label: 'Beta', value: 'beta' },
                      { label: 'Normal', value: 'normal' },
                      { label: 'Linear Quadratic', value: 'linear_quadratic' },
                      { label: 'KL Optimal', value: 'kl_optimal' },
                    ]"
                    class="w-full" 
                  />
                  <p class="text-xs text-muted mt-1">
                    Noise schedule for the denoising process
                  </p>
                </div>
              </div>

              <!-- Model Parameters -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Shift</label>
                    <span class="text-sm text-muted">{{ shift.toFixed(1) }}</span>
                  </div>
                  <USlider v-model="shift" :min="1.0" :max="5.0" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Model sampling shift parameter (affects generation characteristics)
                  </p>
                </div>

                <div>
                  <label class="block text-sm font-medium text-white mb-2">Seed (optional)</label>
                  <UInput v-model="seed" type="number" placeholder="Random" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Leave empty for random generation
                  </p>
                </div>
              </div>
            </div>

            <!-- Generate Button -->
            <SpotlightButton type="submit" :loading="loading" :animate="false" class="w-full py-3">
              <div class="flex items-center justify-center gap-2 relative z-10">
                <UIcon name="i-lucide-image" class="size-4" :class="[loading ? 'animate-pulse' : '']" />
                <span class="text-white font-medium">
                  {{ loading ? 'Generating Image...' : 'Generate Image' }}
                </span>
              </div>
            </SpotlightButton>
          </form>
        </SpotlightCard>

        <!-- Generated Images Display -->
        <SpotlightCard v-if="generatedImages || loading">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">
              Generated Images
            </h3>
            <div v-if="!loading && generatedImages" class="flex items-center gap-2">
              <span class="text-sm text-muted">{{ generatedImages.length }} image{{ generatedImages.length > 1 ? 's' : '' }}</span>
            </div>
          </div>

          <div class="space-y-4">
            <!-- Loading State -->
            <div v-if="loading" class="aspect-square flex items-center justify-center bg-gray-900 rounded-lg">
              <div class="text-center">
                <div
                  class="w-12 h-12 border-4 border-green-400/20 border-t-green-400 rounded-full animate-spin mx-auto mb-4">
                </div>
                <p class="text-white/80">
                  Generating your images...
                </p>
                <p class="text-sm text-muted">
                  This may take a few moments
                </p>
              </div>
            </div>

            <!-- Generated Images Grid -->
            <div v-else-if="generatedImages" class="space-y-6">
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div
                  v-for="(imageUrl, index) in generatedImages"
                  :key="index"
                  class="relative group rounded-lg overflow-hidden bg-gray-900 border border-gray-700"
                >
                  <img :src="imageUrl" :alt="`Generated image ${index + 1}`" class="w-full h-auto" />
                  
                  <!-- Image Overlay -->
                  <div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
                    <div class="flex items-center gap-2">
                      <button
                        class="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                        title="Download image"
                        @click="downloadImage(imageUrl, index)"
                      >
                        <UIcon name="i-lucide-download" class="size-4 text-white" />
                      </button>
                    </div>
                  </div>

                  <!-- Image Index -->
                  <div class="absolute top-2 left-2 px-2 py-1 bg-black/50 rounded text-xs text-white">
                    {{ index + 1 }} / {{ generatedImages.length }}
                  </div>
                </div>
              </div>

              <!-- Generation Info -->
              <div class="bg-gray-900/50 p-4 rounded-lg">
                <h5 class="text-sm font-medium text-white mb-2 flex items-center gap-2">
                  <UIcon name="i-lucide-info" class="size-4 text-gray-400" />
                  Generation Settings
                </h5>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-300">
                  <div>
                    <span class="text-gray-400">Steps:</span> {{ steps }}
                  </div>
                  <div>
                    <span class="text-gray-400">CFG:</span> {{ cfg }}
                  </div>
                  <div>
                    <span class="text-gray-400">Size:</span> {{ width }}×{{ height }}
                  </div>
                  <div>
                    <span class="text-gray-400">Sampler:</span> {{ samplerName }}
                  </div>
                </div>
                <div class="mt-2">
                  <span class="text-gray-400 text-xs">Prompt:</span>
                  <p class="text-gray-300 text-xs mt-1 leading-relaxed">{{ prompt }}</p>
                </div>
              </div>
            </div>
          </div>
        </SpotlightCard>

        <!-- System Metrics and Live Logs -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SpotlightCard>
            <SystemMetrics />
          </SpotlightCard>
          <SpotlightCard>
            <LogStream />
          </SpotlightCard>
        </div>
      </div>
    </div>
  </div>
</template>