<script setup lang="ts">
useHead({
  title: 'AI Text-to-Speech',
  meta: [
    { name: 'description', content: 'Generate speech from text using AI-powered text-to-speech' },
  ],
})

const toast = useToast()
const config = useRuntimeConfig()

// Basic settings (always visible)
const text = ref('')
const loading = ref(false)
const generatedAudio = ref(null)
const generationHistory = ref([])

// Advanced settings (behind toggle)
const showAdvancedSettings = ref(false)
const maxNewTokens = ref(4096)
const flowCfgScale = ref(0.7)
const exaggeration = ref(0.5)
const temperature = ref(0.8)
const cfgWeight = ref(0.5)
const repetitionPenalty = ref(1.2)
const minP = ref(0.05)
const topP = ref(1.0)
const seed = ref()
const useWatermark = ref(false)

// Audio prompt upload
const uploadedAudio = ref(null)
const uploadedAudioUrl = ref('')

// Download audio function
function downloadAudio() {
  if (!generatedAudio.value) return

  const link = document.createElement('a')
  link.href = generatedAudio.value
  link.download = `generated-speech-${Date.now()}.wav`
  link.click()
}

async function generateSpeech() {
  if (loading.value)
    return

  // Validate requirements
  if (!text.value.trim()) {
    toast.add({
      title: 'Text Required',
      description: 'Please enter text to synthesize',
      color: 'red',
    })
    return
  }

  loading.value = true

  try {
    const formData = new FormData()
    formData.append('text', text.value)
    formData.append('max_new_tokens', maxNewTokens.value.toString())
    formData.append('flow_cfg_scale', flowCfgScale.value.toString())
    formData.append('exaggeration', exaggeration.value.toString())
    formData.append('temperature', temperature.value.toString())
    formData.append('cfg_weight', cfgWeight.value.toString())
    formData.append('repetition_penalty', repetitionPenalty.value.toString())
    formData.append('min_p', minP.value.toString())
    formData.append('top_p', topP.value.toString())
    formData.append('seed', (seed.value ? Number.parseInt(seed.value) : -1).toString())
    formData.append('use_watermark', useWatermark.value.toString())
    
    // Add audio file if uploaded
    if (uploadedAudio.value) {
      formData.append('prompt_wav', uploadedAudio.value)
    }

    // Use $fetch like the other endpoints
    const audioBlob = await $fetch(`${config.public.apiBase}/api/tts/generate`, {
      method: 'POST',
      body: formData,
      responseType: 'blob'
    })

    const audioUrl = URL.createObjectURL(audioBlob)

    generatedAudio.value = audioUrl

    const newAudio = {
      src: audioUrl,
      text: text.value,
      temperature: temperature.value,
      maxTokens: maxNewTokens.value,
      timestamp: new Date().toISOString(),
    }

    generationHistory.value.unshift(newAudio)

    toast.add({
      title: 'Success!',
      description: 'Speech generated successfully',
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


// Audio upload handling
function handleAudioUpload(event) {
  const file = event.target.files[0]
  if (!file)
    return

  // Validate file type
  if (!file.type.startsWith('audio/')) {
    toast.add({
      title: 'Invalid File',
      description: 'Please select an audio file',
      color: 'red',
    })
    return
  }

  uploadedAudio.value = file
  uploadedAudioUrl.value = URL.createObjectURL(file)

  toast.add({
    title: 'Audio Uploaded',
    description: 'Voice prompt ready for TTS',
    color: 'green',
  })
}

// Remove uploaded audio
function removeUploadedAudio() {
  if (uploadedAudioUrl.value) {
    URL.revokeObjectURL(uploadedAudioUrl.value)
  }
  uploadedAudio.value = null
  uploadedAudioUrl.value = ''
}

// Cleanup audio URLs on unmount
onUnmounted(() => {
  if (generatedAudio.value) {
    URL.revokeObjectURL(generatedAudio.value)
  }
  if (uploadedAudioUrl.value) {
    URL.revokeObjectURL(uploadedAudioUrl.value)
  }
  generationHistory.value.forEach(audio => {
    if (audio.src) {
      URL.revokeObjectURL(audio.src)
    }
  })
})
</script>

<template>
  <div class="min-h-screen py-12">
    <div class="mx-auto max-w-6xl px-4">
      <!-- Header -->
      <div class="text-center mb-12">
        <div class="flex items-center justify-center mb-6">
          <div class="relative">
            <div
              class="w-16 h-16 rounded-full bg-gradient-to-br from-orange-500/20 to-red-500/20 flex items-center justify-center">
              <UIcon name="i-lucide-mic" class="size-8 text-orange-400" />
            </div>
            <div class="absolute -top-1 -right-1 animate-pulse">
              <div
                class="w-6 h-6 bg-gradient-to-r from-orange-400 to-red-400 rounded-full flex items-center justify-center">
                <UIcon name="i-lucide-sparkles" class="size-3 text-black animate-spin" />
              </div>
            </div>
          </div>
        </div>

        <h1
          class="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-orange-400 via-red-400 to-pink-400 bg-clip-text text-transparent mb-4">
          AI Text-to-Speech
        </h1>

        <p class="text-lg text-muted max-w-2xl mx-auto mb-6">
          Generate natural-sounding speech from text using advanced neural voice synthesis
        </p>

        <div class="flex items-center justify-center gap-6 text-sm">
          <div class="flex items-center gap-2 text-orange-400">
            <UIcon name="i-lucide-type" class="size-4" />
            <span>Text-to-Speech</span>
          </div>
          <div class="flex items-center gap-2 text-red-400">
            <UIcon name="i-lucide-mic" class="size-4" />
            <span>Voice Cloning</span>
          </div>
          <div class="flex items-center gap-2 text-pink-400">
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
              <UIcon name="i-lucide-edit" class="size-5 text-orange-400" />
              Create Your Speech
            </h3>
            <div class="flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-orange-500 animate-pulse"></div>
              <span class="text-xs text-orange-400 font-medium">Ready</span>
            </div>
          </div>

          <form class="space-y-6" @submit.prevent="generateSpeech">
            <!-- Text Input -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Text to Synthesize
                <span class="text-red-400">*</span>
              </label>
              <UTextarea v-model="text"
                placeholder="Hello world! This is a test of the text-to-speech system. Please speak this text in a natural, expressive voice."
                rows="4" class="w-full" />
              <p class="text-xs text-muted mt-1">
                Enter the text you want to convert to speech
              </p>
            </div>

            <!-- Audio Prompt Upload (Optional) -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Voice Prompt (optional)
              </label>
              <p class="text-xs text-muted mb-3">
                Upload an audio sample to clone the voice style and characteristics
              </p>

              <!-- Upload Area -->
              <div v-if="!uploadedAudioUrl"
                class="relative border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-gray-500 transition-colors">
                <input type="file" accept="audio/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  @change="handleAudioUpload" />
                <UIcon name="i-lucide-upload" class="size-8 text-gray-400 mx-auto mb-2" />
                <p class="text-white font-medium mb-1">
                  Click to upload voice sample (optional)
                </p>
                <p class="text-xs text-muted">
                  WAV, MP3, M4A, OGG
                </p>
              </div>

              <!-- Uploaded Audio Preview -->
              <div v-else class="space-y-3">
                <div class="flex items-center gap-3 p-3 bg-gray-800 rounded-lg">
                  <UIcon name="i-lucide-music" class="size-5 text-orange-400" />
                  <div class="flex-1">
                    <p class="text-sm text-white font-medium">Voice prompt uploaded</p>
                    <audio :src="uploadedAudioUrl" controls class="w-full mt-2 h-8">
                      Your browser does not support the audio element.
                    </audio>
                  </div>
                  <button type="button"
                    class="w-8 h-8 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center transition-colors"
                    @click="removeUploadedAudio">
                    <UIcon name="i-lucide-x" class="size-4 text-white" />
                  </button>
                </div>
                <p class="text-sm text-green-400 flex items-center gap-1">
                  <UIcon name="i-lucide-check" class="size-3" />
                  Voice cloning enabled
                </p>
              </div>
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
                Fine-tune voice characteristics and generation parameters
              </p>
            </div>

            <!-- Advanced Settings Panel -->
            <div v-show="showAdvancedSettings" class="space-y-4 bg-gray-900/50 p-4 rounded-lg">
              <!-- Voice Controls -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Voice Exaggeration</label>
                    <span class="text-sm text-muted">{{ exaggeration.toFixed(2) }}</span>
                  </div>
                  <USlider v-model="exaggeration" :min="0" :max="1" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Controls expressiveness and emotion in the voice
                  </p>
                </div>

                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Temperature</label>
                    <span class="text-sm text-muted">{{ temperature.toFixed(2) }}</span>
                  </div>
                  <USlider v-model="temperature" :min="0.1" :max="2.0" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Controls randomness and variety in speech generation
                  </p>
                </div>
              </div>

              <!-- Generation Controls -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Flow CFG Scale</label>
                    <span class="text-sm text-muted">{{ flowCfgScale.toFixed(2) }}</span>
                  </div>
                  <USlider v-model="flowCfgScale" :min="0.1" :max="2.0" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Controls adherence to text prompt
                  </p>
                </div>

                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">CFG Weight</label>
                    <span class="text-sm text-muted">{{ cfgWeight.toFixed(2) }}</span>
                  </div>
                  <USlider v-model="cfgWeight" :min="0" :max="1" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Balances guidance strength
                  </p>
                </div>
              </div>

              <!-- Fine-tuning Controls -->
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Repetition Penalty</label>
                    <span class="text-sm text-muted">{{ repetitionPenalty.toFixed(2) }}</span>
                  </div>
                  <USlider v-model="repetitionPenalty" :min="1.0" :max="2.0" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Reduces repetitive speech
                  </p>
                </div>

                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Min P</label>
                    <span class="text-sm text-muted">{{ minP.toFixed(3) }}</span>
                  </div>
                  <USlider v-model="minP" :min="0" :max="1" :step="0.01" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Minimum probability threshold
                  </p>
                </div>

                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Top P</label>
                    <span class="text-sm text-muted">{{ topP.toFixed(2) }}</span>
                  </div>
                  <USlider v-model="topP" :min="0" :max="1" :step="0.1" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Nucleus sampling threshold
                  </p>
                </div>
              </div>

              <!-- Token and Generation Controls -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div class="flex items-center justify-between mb-2">
                    <label class="block text-sm font-medium text-white">Max New Tokens</label>
                    <span class="text-sm text-muted">{{ maxNewTokens }}</span>
                  </div>
                  <USlider v-model="maxNewTokens" :min="500" :max="4096" :step="100" class="w-full" />
                  <p class="text-xs text-muted mt-1">
                    Maximum number of tokens to generate (constrains output length)
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

              <!-- Additional Settings -->
              <div class="flex items-center gap-3">
                <UCheckbox v-model="useWatermark" />
                <div>
                  <label class="text-sm font-medium text-white">Add Watermark</label>
                  <p class="text-xs text-muted">
                    Embed audio watermark for identification
                  </p>
                </div>
              </div>
            </div>

            <!-- Generate Button -->
            <SpotlightButton type="submit" :loading="loading" :animate="false" class="w-full py-3">
              <div class="flex items-center justify-center gap-2 relative z-10">
                <UIcon name="i-lucide-mic" class="size-4" :class="[loading ? 'animate-pulse' : '']" />
                <span class="text-white font-medium">
                  {{ loading ? 'Generating Speech...' : 'Generate Speech' }}
                </span>
              </div>
            </SpotlightButton>
          </form>
        </SpotlightCard>

        <!-- Generated Audio Display -->
        <SpotlightCard v-if="generatedAudio || loading">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">
              Generated Speech
            </h3>
            <div v-if="!loading && generatedAudio" class="flex items-center gap-2">
              <button class="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Download audio"
                @click="downloadAudio">
                <UIcon name="i-lucide-download" class="size-4 text-gray-400" />
              </button>
            </div>
          </div>

          <div class="space-y-4">
            <!-- Loading State -->
            <div v-if="loading" class="aspect-[3/1] flex items-center justify-center bg-gray-900 rounded-lg">
              <div class="text-center">
                <div
                  class="w-12 h-12 border-4 border-orange-400/20 border-t-orange-400 rounded-full animate-spin mx-auto mb-4">
                </div>
                <p class="text-white/80">
                  Generating your speech...
                </p>
                <p class="text-sm text-muted">
                  This may take a few moments
                </p>
              </div>
            </div>

            <!-- Audio Player -->
            <div v-else-if="generatedAudio" class="space-y-4">
              <div class="bg-gradient-to-r from-orange-500/10 to-red-500/10 p-6 rounded-lg border border-orange-500/20">
                <div class="flex items-center gap-3 mb-4">
                  <UIcon name="i-lucide-music" class="size-5 text-orange-400" />
                  <h4 class="text-white font-medium">Your Generated Speech</h4>
                </div>
                <audio :src="generatedAudio" controls class="w-full">
                  Your browser does not support the audio tag.
                </audio>
              </div>

              <!-- Text Display -->
              <div class="bg-gray-900/50 p-4 rounded-lg">
                <h5 class="text-sm font-medium text-white mb-2 flex items-center gap-2">
                  <UIcon name="i-lucide-type" class="size-4 text-gray-400" />
                  Generated Text
                </h5>
                <p class="text-gray-300 text-sm leading-relaxed">
                  {{ text }}
                </p>
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