<script setup lang="ts">
  interface SystemMetrics {
    cpu: {
      percent: number
      count: number
      frequency?: number
    }
    memory: {
      percent: number
      used_gb: number
      total_gb: number
      available_gb: number
    }
    disk: {
      percent: number
      used_gb: number
      total_gb: number
      free_gb: number
    }
    gpus: Array<{
      id: number
      name: string
      load: number
      memory_used: number
      memory_total: number
      memory_percent: number
      temperature?: number
    }>
    timestamp: number
  }

  const config = useRuntimeConfig()
  const metrics = ref<SystemMetrics | null>(null)
  const isConnected = ref(false)
  const pollingTimer = ref<NodeJS.Timeout | null>(null)
  const isPaused = ref(false)

  function getCpuColor(percent: number): string {
    if (percent < 30)
      return 'bg-green-500'
    if (percent < 70)
      return 'bg-yellow-500'
    return 'bg-red-500'
  }

  function getMemoryColor(percent: number): string {
    if (percent < 50)
      return 'bg-blue-500'
    if (percent < 80)
      return 'bg-yellow-500'
    return 'bg-red-500'
  }

  function getGpuColor(percent: number): string {
    if (percent < 30)
      return 'bg-green-500'
    if (percent < 70)
      return 'bg-yellow-500'
    return 'bg-red-500'
  }

  function getDiskColor(percent: number): string {
    if (percent < 60)
      return 'bg-green-500'
    if (percent < 85)
      return 'bg-yellow-500'
    return 'bg-red-500'
  }

  function togglePause() {
    isPaused.value = !isPaused.value
    if (!isPaused.value) {
      // Resume polling immediately
      fetchMetrics()
    }
  }

  async function fetchMetrics() {
    // Don't fetch if paused
    if (isPaused.value) {
      pollingTimer.value = setTimeout(fetchMetrics, 3000)
      return
    }

    try {
      const response = await $fetch<SystemMetrics>(`${config.public.apiBase}/api/system/metrics`)
      metrics.value = response
      isConnected.value = true
    }
    catch (error) {
      console.error('Failed to fetch system metrics:', error)
      isConnected.value = false
    }

    // Poll every 3 seconds
    pollingTimer.value = setTimeout(fetchMetrics, 3000)
  }

  // Start polling on mount
  onMounted(() => {
    fetchMetrics()
  })

  // Cleanup on unmount
  onBeforeUnmount(() => {
    if (pollingTimer.value) {
      clearTimeout(pollingTimer.value)
    }
  })
</script>

<template>
  <div class="bg-black/50 rounded-lg p-4">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-white font-medium flex items-center gap-2">
        <UIcon name="i-lucide-activity" class="size-4" />
        System Metrics
      </h3>
      <div class="flex items-center gap-2">
        <div class="flex items-center gap-1">
          <div
            class="w-2 h-2 rounded-full"
            :class="[
              isConnected && !isPaused ? 'bg-green-500 animate-pulse' : isPaused ? 'bg-yellow-500' : 'bg-red-500',
            ]"
          ></div>
          <span class="text-xs text-gray-400">
            {{ isPaused ? 'Paused' : isConnected ? 'Live' : 'Offline' }}
          </span>
        </div>
        <button
          class="p-1 hover:bg-white/10 rounded transition-colors"
          :title="isPaused ? 'Resume metrics' : 'Pause metrics'"
          @click="togglePause"
        >
          <UIcon :name="isPaused ? 'i-lucide-play' : 'i-lucide-pause'" class="size-3 text-gray-400" />
        </button>
      </div>
    </div>

    <div v-if="metrics" class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
      <!-- CPU -->
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-gray-300 flex items-center gap-1">
            <UIcon name="i-lucide-cpu" class="size-3" />
            CPU
          </span>
          <span class="text-white font-mono">{{ metrics.cpu.percent }}%</span>
        </div>
        <div class="w-full bg-gray-700 rounded-full h-2">
          <div
            class="h-2 rounded-full transition-all duration-300"
            :class="getCpuColor(metrics.cpu.percent)"
            :style="{ width: `${metrics.cpu.percent}%` }"
          ></div>
        </div>
        <div class="text-xs text-gray-400">
          {{ metrics.cpu.count }} cores
          <span v-if="metrics.cpu.frequency">• {{ metrics.cpu.frequency }} MHz</span>
        </div>
      </div>

      <!-- Memory -->
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-gray-300 flex items-center gap-1">
            <UIcon name="i-lucide-hard-drive" class="size-3" />
            Memory
          </span>
          <span class="text-white font-mono">{{ metrics.memory.percent }}%</span>
        </div>
        <div class="w-full bg-gray-700 rounded-full h-2">
          <div
            class="h-2 rounded-full transition-all duration-300"
            :class="getMemoryColor(metrics.memory.percent)"
            :style="{ width: `${metrics.memory.percent}%` }"
          ></div>
        </div>
        <div class="text-xs text-gray-400">
          {{ metrics.memory.used_gb }}GB / {{ metrics.memory.total_gb }}GB
        </div>
      </div>

      <!-- GPUs -->
      <div v-if="metrics.gpus && metrics.gpus.length > 0" class="md:col-span-2 space-y-3">
        <div v-for="gpu in metrics.gpus" :key="gpu.id" class="space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-gray-300 flex items-center gap-1">
              <UIcon name="i-lucide-zap" class="size-3" />
              {{ gpu.name }}
            </span>
            <div class="flex items-center gap-3 text-xs">
              <span class="text-white font-mono">{{ gpu.load }}%</span>
              <span class="text-gray-400">{{ gpu.memory_used }}MB / {{ gpu.memory_total }}MB</span>
              <span v-if="gpu.temperature" class="text-yellow-400">{{ gpu.temperature }}°C</span>
            </div>
          </div>

          <!-- GPU Load -->
          <div class="space-y-1">
            <div class="flex justify-between text-xs">
              <span class="text-gray-400">GPU Load</span>
              <span class="text-white">{{ gpu.load }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-1.5">
              <div
                class="h-1.5 rounded-full transition-all duration-300"
                :class="getGpuColor(gpu.load)"
                :style="{ width: `${gpu.load}%` }"
              ></div>
            </div>
          </div>

          <!-- GPU Memory -->
          <div class="space-y-1">
            <div class="flex justify-between text-xs">
              <span class="text-gray-400">VRAM</span>
              <span class="text-white">{{ gpu.memory_percent }}%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-1.5">
              <div
                class="h-1.5 rounded-full transition-all duration-300"
                :class="getMemoryColor(gpu.memory_percent)"
                :style="{ width: `${gpu.memory_percent}%` }"
              ></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Disk -->
      <div class="md:col-span-2 space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-gray-300 flex items-center gap-1">
            <UIcon name="i-lucide-database" class="size-3" />
            Disk
          </span>
          <span class="text-white font-mono">{{ metrics.disk.percent }}%</span>
        </div>
        <div class="w-full bg-gray-700 rounded-full h-2">
          <div
            class="h-2 rounded-full transition-all duration-300"
            :class="getDiskColor(metrics.disk.percent)"
            :style="{ width: `${metrics.disk.percent}%` }"
          ></div>
        </div>
        <div class="text-xs text-gray-400">
          {{ metrics.disk.used_gb }}GB used • {{ metrics.disk.free_gb }}GB free • {{ metrics.disk.total_gb }}GB total
        </div>
      </div>
    </div>

    <div v-else-if="!isConnected" class="text-center text-gray-500 py-4">
      Unable to load system metrics
    </div>

    <div v-else class="text-center text-gray-500 py-4">
      Loading metrics...
    </div>
  </div>
</template>

<style scoped>
/* Custom styles for metrics bars */
.transition-all {
  transition: width 0.3s ease-in-out;
}
</style>
