<script setup lang="ts">
interface LogEntry {
  timestamp: string
  level: string
  message: string
  heartbeat?: boolean
}

const config = useRuntimeConfig()
const logs = ref<LogEntry[]>([])
const isConnected = ref(false)
const logsContainer = ref<HTMLElement>()
const sessionId = ref<string>('')
const pollingTimer = ref<NodeJS.Timeout | null>(null)
const isPaused = ref(false)

function getLogLevelColor(level: string): string {
  switch (level) {
    case 'error': return 'bg-red-500'
    case 'warning': return 'bg-yellow-500'
    case 'success': return 'bg-green-500'
    case 'system': return 'bg-blue-500'
    default: return 'bg-gray-400'
  }
}

function getLogTextColor(level: string): string {
  switch (level) {
    case 'error': return 'text-red-400'
    case 'warning': return 'text-yellow-400'
    case 'success': return 'text-green-400'
    case 'system': return 'text-blue-400'
    default: return 'text-gray-300'
  }
}

function clearLogs() {
  logs.value = []
}

function togglePause() {
  isPaused.value = !isPaused.value
  if (!isPaused.value) {
    // Resume polling immediately
    pollLogs()
  }
}

let logIndex = 0

async function connectToLogs() {
  isConnected.value = true
  pollLogs()
}

async function pollLogs() {
  // Don't poll if paused
  if (isPaused.value) {
    pollingTimer.value = setTimeout(pollLogs, 1000)
    return
  }

  try {
    const response = await $fetch<{ logs: LogEntry[], total: number }>(`${config.public.apiBase}/api/logs?since=${logIndex}`)

    if (response.logs && response.logs.length > 0) {
      // Add only new logs
      logs.value.push(...response.logs)
      logIndex = response.total

      // Limit logs to prevent memory issues
      if (logs.value.length > 1000) {
        logs.value = logs.value.slice(-500)
      }
    }

    isConnected.value = true
  }
  catch (error) {
    console.error('Failed to fetch logs:', error)
    isConnected.value = false
  }

  // Poll every 1 second
  pollingTimer.value = setTimeout(pollLogs, 1000)
}

// Connect on mount
onMounted(() => {
  connectToLogs()
})

// Cleanup on unmount
onBeforeUnmount(() => {
  if (pollingTimer.value) {
    clearTimeout(pollingTimer.value)
  }
})
</script>

<template>
  <div class="bg-black/50 rounded-lg p-4 font-mono text-sm max-h-64 overflow-y-auto">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-white font-medium flex items-center gap-2">
        <UIcon name="i-lucide-terminal" class="size-4" />
        Live Logs
      </h3>
      <div class="flex items-center gap-2">
        <div class="flex items-center gap-1">
          <div class="w-2 h-2 rounded-full" :class="[
            isConnected && !isPaused ? 'bg-green-500 animate-pulse' : isPaused ? 'bg-yellow-500' : 'bg-red-500',
          ]"></div>
          <span class="text-xs text-gray-400">
            {{ isPaused ? 'Paused' : isConnected ? 'Live' : 'Disconnected' }}
          </span>
        </div>
        <button class="p-1 hover:bg-white/10 rounded transition-colors" :title="isPaused ? 'Resume logs' : 'Pause logs'"
          @click="togglePause">
          <UIcon :name="isPaused ? 'i-lucide-play' : 'i-lucide-pause'" class="size-3 text-gray-400" />
        </button>
        <button class="p-1 hover:bg-white/10 rounded transition-colors" title="Clear logs" @click="clearLogs">
          <UIcon name="i-lucide-trash-2" class="size-3 text-gray-400" />
        </button>
      </div>
    </div>

    <div ref="logsContainer" class="space-y-1 flex flex-col-reverse">
      <div v-for="(log, index) in logs" :key="logs.length - index" class="flex items-start gap-2 text-xs">
        <span class="text-gray-500 shrink-0">{{ log.timestamp }}</span>
        <span class="shrink-0 w-1 h-1 rounded-full mt-2" :class="[
          getLogLevelColor(log.level),
        ]"></span>
        <span class="break-words" :class="[
          getLogTextColor(log.level),
        ]">{{ log.message }}</span>
      </div>

      <div v-if="logs.length === 0" class="text-gray-500 text-center py-4">
        Waiting for logs...
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Custom scrollbar for dark theme */
.overflow-y-auto::-webkit-scrollbar {
  width: 4px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 2px;
}

.overflow-y-auto::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}
</style>
