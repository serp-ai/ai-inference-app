<script setup lang="ts">
useHead({
  title: 'Bulk Content Creator',
  meta: [
    { name: 'description', content: 'Create and manage AI content templates for bulk generation' },
  ],
})

import Papa from 'papaparse'

const toast = useToast()
const config = useRuntimeConfig()

// Main application state
const currentTab = ref('templates') // templates, generate, history
const templates = ref([])
const generationHistory = ref([])

// Template creation state
const isCreatingTemplate = ref(false)
const templateForm = ref({
  name: '',
  description: '',
  instructions: '',
  generatedPrompt: '',
  variables: [] as Array<{ name: string, type: string, description: string, required: boolean }>,
  isEditing: false,
  editingIndex: -1
})

// Generation state
const selectedTemplate = ref(null)
const generationMode = ref('single') // single, batch, csv
const singleGenVars = ref({})
const batchGenVars = ref([{}])
const csvData = ref([])
const csvFile = ref(null)
const isGenerating = ref(false)
const generationProgress = ref({ current: 0, total: 0, results: [] })

// Template AI generation
const isGeneratingTemplate = ref(false)

// Persistent storage functions
const STORAGE_KEYS = {
  TEMPLATES: 'bulk_creator_templates',
  HISTORY: 'bulk_creator_history'
}

function saveToStorage() {
  try {
    localStorage.setItem(STORAGE_KEYS.TEMPLATES, JSON.stringify(templates.value))
    localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(generationHistory.value))
  } catch (error) {
    console.warn('Failed to save to localStorage:', error)
  }
}

function loadFromStorage() {
  try {
    const savedTemplates = localStorage.getItem(STORAGE_KEYS.TEMPLATES)
    const savedHistory = localStorage.getItem(STORAGE_KEYS.HISTORY)

    if (savedTemplates) {
      templates.value = JSON.parse(savedTemplates)
    }
    if (savedHistory) {
      generationHistory.value = JSON.parse(savedHistory)
    }
  } catch (error) {
    console.warn('Failed to load from localStorage:', error)
  }
}

// Initialize storage on mount
onMounted(() => {
  loadFromStorage()
})

// Watch for changes and save
watch([templates, generationHistory], () => {
  saveToStorage()
}, { deep: true })

// Template management functions
async function generatePromptTemplate() {
  if (!templateForm.value.instructions.trim()) {
    toast.add({
      title: 'Instructions Required',
      description: 'Please provide instructions for what the AI should create',
      color: 'red',
    })
    return
  }

  isGeneratingTemplate.value = true

  try {
    const systemPrompt = `You are an AI prompt template generator. Your task is to create a prompt template based on the user's instructions.

Rules:
1. Create a clear, detailed prompt template that will generate consistent, high-quality content
2. Include template variables in the format {{variable_name}} where customization is needed
3. Make the prompt specific and actionable
4. Consider the context and requirements provided
5. The template should be ready to use with an LLM

Output format should be just the prompt template, nothing else.`

    const response = await $fetch(`${config.public.apiBase}/api/llm/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: {
        model_name_or_path: 'Qwen/Qwen3-4B-Thinking-2507-FP8', // Default to a fast model
        prompt: `Create a prompt template for: ${templateForm.value.instructions}`,
        system_prompt: systemPrompt,
        use_local: true,
        model_base_url: 'https://api.openai.com/v1',
        temperature: 0.7,
        max_length: 1024,
      }
    })

    templateForm.value.generatedPrompt = response.response.trim()

    // Auto-detect variables in the generated prompt
    const variableMatches = templateForm.value.generatedPrompt.match(/\{\{([^}]+)\}\}/g) || []
    const detectedVars = variableMatches.map(match => {
      const name = match.replace(/[{}]/g, '')
      return {
        name: name,
        type: 'text',
        description: `Enter ${name.replace(/_/g, ' ')}`,
        required: true
      }
    })

    // Remove duplicates
    const uniqueVars = detectedVars.filter((v, i, arr) =>
      arr.findIndex(x => x.name === v.name) === i
    )

    templateForm.value.variables = uniqueVars

    toast.add({
      title: 'Template Generated!',
      description: `Created template with ${uniqueVars.length} variables`,
      color: 'green',
    })
  } catch (error) {
    toast.add({
      title: 'Template Generation Failed',
      description: error.message || 'Something went wrong',
      color: 'red',
    })
  } finally {
    isGeneratingTemplate.value = false
  }
}

function saveTemplate() {
  if (!templateForm.value.name.trim()) {
    toast.add({
      title: 'Template Name Required',
      description: 'Please enter a name for your template',
      color: 'red',
    })
    return
  }

  if (!templateForm.value.generatedPrompt.trim()) {
    toast.add({
      title: 'Prompt Template Required',
      description: 'Please generate or enter a prompt template',
      color: 'red',
    })
    return
  }

  const template = {
    id: templateForm.value.isEditing ? templates.value[templateForm.value.editingIndex].id : Date.now(),
    name: templateForm.value.name,
    description: templateForm.value.description,
    instructions: templateForm.value.instructions,
    prompt: templateForm.value.generatedPrompt,
    variables: templateForm.value.variables,
    createdAt: templateForm.value.isEditing ? templates.value[templateForm.value.editingIndex].createdAt : new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }

  if (templateForm.value.isEditing) {
    templates.value[templateForm.value.editingIndex] = template
    toast.add({
      title: 'Template Updated',
      description: 'Your template has been updated successfully',
      color: 'green',
    })
  } else {
    templates.value.unshift(template)
    toast.add({
      title: 'Template Saved',
      description: 'Your template has been saved successfully',
      color: 'green',
    })
  }

  resetTemplateForm()
}

function editTemplate(index) {
  const template = templates.value[index]
  templateForm.value = {
    name: template.name,
    description: template.description,
    instructions: template.instructions,
    generatedPrompt: template.prompt,
    variables: [...template.variables],
    isEditing: true,
    editingIndex: index
  }
  isCreatingTemplate.value = true
}

function deleteTemplate(index) {
  const template = templates.value[index]
  if (confirm(`Are you sure you want to delete "${template.name}"?`)) {
    templates.value.splice(index, 1)
    toast.add({
      title: 'Template Deleted',
      description: 'Template has been removed',
      color: 'orange',
    })
  }
}

function resetTemplateForm() {
  templateForm.value = {
    name: '',
    description: '',
    instructions: '',
    generatedPrompt: '',
    variables: [],
    isEditing: false,
    editingIndex: -1
  }
  isCreatingTemplate.value = false
}

function addVariable() {
  templateForm.value.variables.push({
    name: '',
    type: 'text',
    description: '',
    required: true
  })
}

function removeVariable(index) {
  templateForm.value.variables.splice(index, 1)
}

// Generation functions
function selectTemplate(template) {
  selectedTemplate.value = template
  currentTab.value = 'generate'

  // Initialize variables for single generation
  singleGenVars.value = {}
  template.variables.forEach(variable => {
    singleGenVars.value[variable.name] = ''
  })

  // Initialize batch generation with one set
  batchGenVars.value = [{}]
  template.variables.forEach(variable => {
    batchGenVars.value[0][variable.name] = ''
  })
}

function addBatchSet() {
  const newSet = {}
  selectedTemplate.value.variables.forEach(variable => {
    newSet[variable.name] = ''
  })
  batchGenVars.value.push(newSet)
}

function removeBatchSet(index) {
  if (batchGenVars.value.length > 1) {
    batchGenVars.value.splice(index, 1)
  }
}

// CSV handling functions
function handleCSVUpload(event) {
  const file = event.target.files[0]
  if (!file) return

  if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
    toast.add({
      title: 'Invalid File',
      description: 'Please select a CSV file',
      color: 'red',
    })
    return
  }

  csvFile.value = file

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      console.log('Papa Parse Results:', results)
      
      if (results.errors.length > 0) {
        console.error('Papa Parse Errors:', results.errors)
        
        // Check for fatal errors only - ignore delimiter detection warnings
        const fatalErrors = results.errors.filter(error => 
          error.code !== 'UndetectableDelimiter' && 
          error.type !== 'Delimiter'
        )
        
        if (fatalErrors.length > 0) {
          toast.add({
            title: 'CSV Parse Error',
            description: `Error: ${fatalErrors[0].message || 'Unknown parsing error'}`,
            color: 'red',
          })
          return
        }
        
        // If only delimiter warnings, show a friendly message but continue
        if (results.errors.some(e => e.code === 'UndetectableDelimiter')) {
          console.log('Note: Single column CSV detected (no delimiters found)')
        }
      }

      if (!results.data || results.data.length === 0) {
        toast.add({
          title: 'Empty CSV',
          description: 'No data found in CSV file',
          color: 'red',
        })
        return
      }

      // Filter out completely empty rows
      csvData.value = results.data.filter(row => 
        Object.values(row).some(value => value && value.toString().trim())
      )

      if (csvData.value.length === 0) {
        toast.add({
          title: 'No Valid Data',
          description: 'CSV contains no valid data rows',
          color: 'red',
        })
        return
      }

      // Check if required template variables exist as columns
      const csvColumns = Object.keys(csvData.value[0] || {})
      const requiredVars = selectedTemplate.value.variables.filter(v => v.required).map(v => v.name)
      const missingColumns = requiredVars.filter(varName => !csvColumns.includes(varName))
      
      console.log('CSV Columns found:', csvColumns)
      console.log('Required variables:', requiredVars)
      console.log('Missing columns:', missingColumns)
      
      if (missingColumns.length > 0) {
        toast.add({
          title: 'Missing Required Columns',
          description: `CSV columns found: [${csvColumns.join(', ')}]. Missing required: [${missingColumns.join(', ')}]`,
          color: 'orange',
        })
      }

      toast.add({
        title: 'CSV Loaded',
        description: `Loaded ${csvData.value.length} rows from CSV with columns: ${csvColumns.join(', ')}`,
        color: 'green',
      })
    },
    error: (error) => {
      console.error('Papa Parse Error:', error)
      toast.add({
        title: 'CSV Load Failed',
        description: error.message,
        color: 'red',
      })
    }
  })
}

// Content generation functions
async function generateContent(prompt, variables) {
  let finalPrompt = prompt

  // Replace variables in the prompt
  selectedTemplate.value.variables.forEach(variable => {
    const value = variables[variable.name] || ''
    finalPrompt = finalPrompt.replace(new RegExp(`\\{\\{${variable.name}\\}\\}`, 'g'), value)
  })

  const response = await $fetch(`${config.public.apiBase}/api/llm/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: {
      model_name_or_path: 'Qwen/Qwen3-4B-Thinking-2507-FP8',
      prompt: finalPrompt,
      use_local: true,
      model_base_url: 'https://api.openai.com/v1',
      temperature: 0.7,
      max_length: 2048,
    }
  })

  return response.response
}

async function startGeneration() {
  if (!selectedTemplate.value) {
    toast.add({
      title: 'No Template Selected',
      description: 'Please select a template first',
      color: 'red',
    })
    return
  }

  let generationSets = []

  if (generationMode.value === 'single') {
    generationSets = [singleGenVars.value]
  } else if (generationMode.value === 'batch') {
    generationSets = batchGenVars.value
  } else if (generationMode.value === 'csv') {
    generationSets = csvData.value
  }

  // Validate required variables
  const invalidSets = generationSets.filter(set => {
    return selectedTemplate.value.variables.some(variable => {
      if (variable.required && (!set[variable.name] || !set[variable.name].toString().trim())) {
        return true
      }
      return false
    })
  })

  if (invalidSets.length > 0) {
    toast.add({
      title: 'Missing Required Variables',
      description: `${invalidSets.length} set(s) are missing required variables`,
      color: 'red',
    })
    return
  }

  isGenerating.value = true
  generationProgress.value = {
    current: 0,
    total: generationSets.length,
    results: []
  }

  try {
    for (let i = 0; i < generationSets.length; i++) {
      try {
        const content = await generateContent(selectedTemplate.value.prompt, generationSets[i])

        const result = {
          id: Date.now() + i,
          templateId: selectedTemplate.value.id,
          templateName: selectedTemplate.value.name,
          variables: { ...generationSets[i] },
          content: content,
          createdAt: new Date().toISOString(),
          status: 'success'
        }

        generationProgress.value.results.push(result)
        generationHistory.value.unshift(result)

      } catch (error) {
        const result = {
          id: Date.now() + i,
          templateId: selectedTemplate.value.id,
          templateName: selectedTemplate.value.name,
          variables: { ...generationSets[i] },
          content: null,
          error: error.message,
          createdAt: new Date().toISOString(),
          status: 'failed'
        }

        generationProgress.value.results.push(result)
        generationHistory.value.unshift(result)
      }

      generationProgress.value.current = i + 1

      // Add small delay to prevent overwhelming the API
      if (i < generationSets.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000))
      }
    }

    const successCount = generationProgress.value.results.filter(r => r.status === 'success').length
    const failCount = generationProgress.value.results.filter(r => r.status === 'failed').length

    toast.add({
      title: 'Generation Complete',
      description: `Generated ${successCount} content pieces${failCount > 0 ? `, ${failCount} failed` : ''}`,
      color: successCount > 0 ? 'green' : 'red',
    })

  } catch (error) {
    toast.add({
      title: 'Generation Failed',
      description: error.message || 'Something went wrong',
      color: 'red',
    })
  } finally {
    isGenerating.value = false
  }
}

function downloadResults() {
  if (generationProgress.value.results.length === 0) return

  const results = generationProgress.value.results.map(result => ({
    template: result.templateName,
    status: result.status,
    content: result.content || result.error,
    variables: JSON.stringify(result.variables),
    created_at: result.createdAt
  }))

  const csv = Papa.unparse(results)
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = url
  link.download = `bulk-generation-results-${Date.now()}.csv`
  link.click()

  URL.revokeObjectURL(url)
}

function clearHistory() {
  if (confirm('Are you sure you want to clear all generation history?')) {
    generationHistory.value = []
    toast.add({
      title: 'History Cleared',
      description: 'All generation history has been cleared',
      color: 'orange',
    })
  }
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    toast.add({
      title: 'Copied!',
      description: 'Content copied to clipboard',
      color: 'green',
    })
  }).catch(() => {
    toast.add({
      title: 'Copy Failed',
      description: 'Failed to copy to clipboard',
      color: 'red',
    })
  })
}
</script>

<template>
  <div class="min-h-screen py-12">
    <div class="mx-auto max-w-7xl px-4">
      <!-- Header -->
      <div class="text-center mb-12">
        <div class="flex items-center justify-center mb-6">
          <div class="relative">
            <div
              class="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center">
              <UIcon name="i-lucide-factory" class="size-8 text-purple-400" />
            </div>
            <div class="absolute -top-1 -right-1 animate-pulse">
              <div
                class="w-6 h-6 bg-gradient-to-r from-purple-400 to-blue-400 rounded-full flex items-center justify-center">
                <UIcon name="i-lucide-sparkles" class="size-3 text-black animate-spin" />
              </div>
            </div>
          </div>
        </div>

        <h1
          class="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent mb-4">
          Bulk Content Creator
        </h1>

        <p class="text-lg text-muted max-w-3xl mx-auto mb-6">
          Create AI-powered content templates and generate content at scale with CSV uploads and batch processing
        </p>

        <div class="flex items-center justify-center gap-6 text-sm">
          <div class="flex items-center gap-2 text-purple-400">
            <UIcon name="i-lucide-template" class="size-4" />
            <span>Template System</span>
          </div>
          <div class="flex items-center gap-2 text-blue-400">
            <UIcon name="i-lucide-upload" class="size-4" />
            <span>CSV Import</span>
          </div>
          <div class="flex items-center gap-2 text-cyan-400">
            <UIcon name="i-lucide-zap" class="size-4" />
            <span>Bulk Generation</span>
          </div>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <div class="flex justify-center mb-8">
        <div class="flex bg-gray-800 rounded-lg p-1">
          <button @click="currentTab = 'templates'"
            class="px-6 py-2 rounded-md transition-all duration-200 flex items-center gap-2"
            :class="currentTab === 'templates' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'">
            <UIcon name="i-lucide-template" class="size-4" />
            Templates
          </button>
          <button @click="currentTab = 'generate'"
            class="px-6 py-2 rounded-md transition-all duration-200 flex items-center gap-2"
            :class="currentTab === 'generate' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'">
            <UIcon name="i-lucide-zap" class="size-4" />
            Generate
          </button>
          <button @click="currentTab = 'history'"
            class="px-6 py-2 rounded-md transition-all duration-200 flex items-center gap-2"
            :class="currentTab === 'history' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'">
            <UIcon name="i-lucide-history" class="size-4" />
            History
          </button>
        </div>
      </div>

      <!-- Templates Tab -->
      <div v-if="currentTab === 'templates'" class="space-y-6">
        <!-- Templates List -->
        <SpotlightCard v-if="!isCreatingTemplate">
          <div class="flex items-center justify-between mb-6">
            <h3 class="text-xl font-semibold text-white flex items-center gap-2">
              <UIcon name="i-lucide-template" class="size-5 text-purple-400" />
              Content Templates
            </h3>
            <SpotlightButton @click="isCreatingTemplate = true" :animate="false">
              <div class="flex items-center gap-2">
                <UIcon name="i-lucide-plus" class="size-4" />
                New Template
              </div>
            </SpotlightButton>
          </div>

          <div v-if="templates.length === 0" class="text-center py-12">
            <div class="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <UIcon name="i-lucide-template" class="size-8 text-gray-400" />
            </div>
            <h4 class="text-lg font-medium text-white mb-2">No Templates Yet</h4>
            <p class="text-muted mb-4">Create your first template to get started with bulk content generation</p>
            <SpotlightButton @click="isCreatingTemplate = true" :animate="false">
              <div class="flex items-center gap-2">
                <UIcon name="i-lucide-plus" class="size-4" />
                Create Template
              </div>
            </SpotlightButton>
          </div>

          <div v-else class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div v-for="(template, index) in templates" :key="template.id"
              class="p-4 bg-gray-800/50 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
              <div class="flex items-start justify-between mb-3">
                <div class="flex-1">
                  <h4 class="text-lg font-medium text-white mb-1">{{ template.name }}</h4>
                  <p class="text-sm text-gray-400 mb-2">{{ template.description || 'No description' }}</p>
                  <div class="flex items-center gap-4 text-xs text-gray-500">
                    <span>{{ template.variables.length }} variables</span>
                    <span>{{ new Date(template.createdAt).toLocaleDateString() }}</span>
                  </div>
                </div>
                <div class="flex items-center gap-1 ml-4">
                  <button @click="selectTemplate(template)"
                    class="p-2 text-green-400 hover:bg-green-400/10 rounded transition-colors" title="Use Template">
                    <UIcon name="i-lucide-play" class="size-4" />
                  </button>
                  <button @click="editTemplate(index)"
                    class="p-2 text-blue-400 hover:bg-blue-400/10 rounded transition-colors" title="Edit Template">
                    <UIcon name="i-lucide-edit" class="size-4" />
                  </button>
                  <button @click="deleteTemplate(index)"
                    class="p-2 text-red-400 hover:bg-red-400/10 rounded transition-colors" title="Delete Template">
                    <UIcon name="i-lucide-trash-2" class="size-4" />
                  </button>
                </div>
              </div>

              <!-- Variables Preview -->
              <div v-if="template.variables.length > 0" class="mt-3">
                <div class="flex flex-wrap gap-1">
                  <span v-for="variable in template.variables.slice(0, 3)" :key="variable.name"
                    class="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded">
                    {{ variable.name }}
                  </span>
                  <span v-if="template.variables.length > 3"
                    class="px-2 py-1 bg-gray-500/20 text-gray-400 text-xs rounded">
                    +{{ template.variables.length - 3 }} more
                  </span>
                </div>
              </div>
            </div>
          </div>
        </SpotlightCard>

        <!-- Template Creation/Edit Form -->
        <SpotlightCard v-if="isCreatingTemplate">
          <div class="flex items-center justify-between mb-6">
            <h3 class="text-xl font-semibold text-white flex items-center gap-2">
              <UIcon name="i-lucide-edit" class="size-5 text-purple-400" />
              {{ templateForm.isEditing ? 'Edit Template' : 'Create New Template' }}
            </h3>
            <button @click="resetTemplateForm" class="p-2 text-gray-400 hover:text-white transition-colors">
              <UIcon name="i-lucide-x" class="size-5" />
            </button>
          </div>

          <form @submit.prevent="saveTemplate" class="space-y-6">
            <!-- Template Details -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium text-white mb-2">
                  Template Name <span class="text-red-400">*</span>
                </label>
                <UInput v-model="templateForm.name" placeholder="My Content Template" class="w-full" />
              </div>
              <div>
                <label class="block text-sm font-medium text-white mb-2">Description</label>
                <UInput v-model="templateForm.description" placeholder="What this template creates..." class="w-full" />
              </div>
            </div>

            <!-- Instructions for AI -->
            <div>
              <label class="block text-sm font-medium text-white mb-2">
                Instructions for AI <span class="text-red-400">*</span>
              </label>
              <UTextarea v-model="templateForm.instructions"
                placeholder="Tell the AI what kind of content to create. Be specific about tone, style, format, and requirements..."
                rows="4" class="w-full" />
              <div class="flex justify-end mt-2">
                <SpotlightButton @click="generatePromptTemplate" :loading="isGeneratingTemplate" :animate="false"
                  type="button">
                  <div class="flex items-center gap-2">
                    <UIcon name="i-lucide-wand-2" class="size-4" />
                    {{ isGeneratingTemplate ? 'Generating...' : 'Generate Template' }}
                  </div>
                </SpotlightButton>
              </div>
            </div>

            <!-- Generated Prompt Template -->
            <div v-if="templateForm.generatedPrompt">
              <label class="block text-sm font-medium text-white mb-2">
                Generated Prompt Template
              </label>
              <UTextarea v-model="templateForm.generatedPrompt" rows="6" class="w-full font-mono"
                placeholder="Your generated template will appear here..." />
              <p class="text-xs text-muted mt-1">
                Use {<!-- -->{variable_name}} syntax for template variables. These will be detected automatically.
              </p>
            </div>

            <!-- Template Variables -->
            <div v-if="templateForm.variables.length > 0">
              <div class="flex items-center justify-between mb-3">
                <label class="block text-sm font-medium text-white">Template Variables</label>
                <button type="button" @click="addVariable"
                  class="text-sm text-purple-400 hover:text-purple-300 flex items-center gap-1">
                  <UIcon name="i-lucide-plus" class="size-3" />
                  Add Variable
                </button>
              </div>

              <div class="space-y-3">
                <div v-for="(variable, index) in templateForm.variables" :key="index"
                  class="grid grid-cols-1 md:grid-cols-4 gap-3 p-3 bg-gray-800/50 rounded border border-gray-700">
                  <div>
                    <label class="block text-xs font-medium text-gray-400 mb-1">Variable Name</label>
                    <UInput v-model="variable.name" placeholder="variable_name" class="w-full" />
                  </div>
                  <div>
                    <label class="block text-xs font-medium text-gray-400 mb-1">Type</label>
                    <USelect v-model="variable.type" :options="[
                      { label: 'Text', value: 'text' },
                      { label: 'Textarea', value: 'textarea' },
                      { label: 'Number', value: 'number' }
                    ]" class="w-full" />
                  </div>
                  <div>
                    <label class="block text-xs font-medium text-gray-400 mb-1">Description</label>
                    <UInput v-model="variable.description" placeholder="Describe this field..." class="w-full" />
                  </div>
                  <div class="flex items-end gap-2">
                    <div class="flex items-center gap-2 mb-1">
                      <UCheckbox v-model="variable.required" />
                      <label class="text-xs text-gray-400">Required</label>
                    </div>
                    <button type="button" @click="removeVariable(index)"
                      class="p-2 text-red-400 hover:bg-red-400/10 rounded transition-colors">
                      <UIcon name="i-lucide-trash-2" class="size-3" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- Action Buttons -->
            <div class="flex items-center gap-3 pt-4">
              <SpotlightButton type="submit" :animate="false" class="flex-1">
                <div class="flex items-center justify-center gap-2">
                  <UIcon name="i-lucide-save" class="size-4" />
                  {{ templateForm.isEditing ? 'Update Template' : 'Save Template' }}
                </div>
              </SpotlightButton>
              <button type="button" @click="resetTemplateForm"
                class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors">
                Cancel
              </button>
            </div>
          </form>
        </SpotlightCard>
      </div>

      <!-- Generate Tab -->
      <div v-if="currentTab === 'generate'" class="space-y-6">
        <!-- Template Selection -->
        <SpotlightCard v-if="!selectedTemplate">
          <div class="text-center py-12">
            <div class="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <UIcon name="i-lucide-template" class="size-8 text-gray-400" />
            </div>
            <h4 class="text-lg font-medium text-white mb-2">Select a Template</h4>
            <p class="text-muted mb-6">Choose a template to start generating content</p>

            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
              <button v-for="template in templates" :key="template.id" @click="selectTemplate(template)"
                class="p-4 bg-gray-800/50 hover:bg-gray-800 rounded-lg border border-gray-700 hover:border-purple-500 transition-all text-left group">
                <h5 class="text-white font-medium mb-2 group-hover:text-purple-300 transition-colors">
                  {{ template.name }}
                </h5>
                <p class="text-sm text-gray-400 mb-3">{{ template.description || 'No description' }}</p>
                <div class="flex items-center justify-between text-xs text-gray-500">
                  <span>{{ template.variables.length }} variables</span>
                  <UIcon name="i-lucide-arrow-right" class="size-4 group-hover:text-purple-400 transition-colors" />
                </div>
              </button>
            </div>

            <div v-if="templates.length === 0" class="mt-6">
              <button @click="currentTab = 'templates'" class="text-purple-400 hover:text-purple-300 underline">
                Create your first template
              </button>
            </div>
          </div>
        </SpotlightCard>

        <!-- Generation Interface -->
        <div v-if="selectedTemplate" class="space-y-6">
          <!-- Selected Template Info -->
          <SpotlightCard>
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-xl font-semibold text-white mb-1">{{ selectedTemplate.name }}</h3>
                <p class="text-muted">{{ selectedTemplate.description || 'No description' }}</p>
              </div>
              <button @click="selectedTemplate = null" class="p-2 text-gray-400 hover:text-white transition-colors"
                title="Change Template">
                <UIcon name="i-lucide-x" class="size-5" />
              </button>
            </div>
          </SpotlightCard>

          <!-- Generation Mode Selection -->
          <SpotlightCard>
            <h4 class="text-lg font-medium text-white mb-4">Generation Mode</h4>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button @click="generationMode = 'single'" class="p-4 rounded-lg border transition-all text-left"
                :class="generationMode === 'single' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700 bg-gray-800/50 hover:bg-gray-800'">
                <div class="flex items-center gap-3 mb-2">
                  <UIcon name="i-lucide-file" class="size-5"
                    :class="generationMode === 'single' ? 'text-purple-400' : 'text-gray-400'" />
                  <span class="font-medium" :class="generationMode === 'single' ? 'text-white' : 'text-gray-300'">Single
                    Generation</span>
                </div>
                <p class="text-sm text-gray-400">Generate one piece of content</p>
              </button>

              <button @click="generationMode = 'batch'" class="p-4 rounded-lg border transition-all text-left"
                :class="generationMode === 'batch' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700 bg-gray-800/50 hover:bg-gray-800'">
                <div class="flex items-center gap-3 mb-2">
                  <UIcon name="i-lucide-files" class="size-5"
                    :class="generationMode === 'batch' ? 'text-purple-400' : 'text-gray-400'" />
                  <span class="font-medium" :class="generationMode === 'batch' ? 'text-white' : 'text-gray-300'">Batch
                    Generation</span>
                </div>
                <p class="text-sm text-gray-400">Generate multiple variations</p>
              </button>

              <button @click="generationMode = 'csv'" class="p-4 rounded-lg border transition-all text-left"
                :class="generationMode === 'csv' ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700 bg-gray-800/50 hover:bg-gray-800'">
                <div class="flex items-center gap-3 mb-2">
                  <UIcon name="i-lucide-upload" class="size-5"
                    :class="generationMode === 'csv' ? 'text-purple-400' : 'text-gray-400'" />
                  <span class="font-medium" :class="generationMode === 'csv' ? 'text-white' : 'text-gray-300'">CSV
                    Import</span>
                </div>
                <p class="text-sm text-gray-400">Bulk generate from CSV</p>
              </button>
            </div>
          </SpotlightCard>

          <!-- Single Generation -->
          <SpotlightCard v-if="generationMode === 'single'">
            <h4 class="text-lg font-medium text-white mb-4">Single Generation</h4>
            <div class="space-y-4">
              <div v-for="variable in selectedTemplate.variables" :key="variable.name">
                <label class="block text-sm font-medium text-white mb-2">
                  {{ variable.name.replace(/_/g, ' ') }}
                  <span v-if="variable.required" class="text-red-400">*</span>
                </label>
                <UTextarea v-if="variable.type === 'textarea'" v-model="singleGenVars[variable.name]"
                  :placeholder="variable.description" rows="3" class="w-full" />
                <UInput v-else v-model="singleGenVars[variable.name]"
                  :type="variable.type === 'number' ? 'number' : 'text'" :placeholder="variable.description"
                  class="w-full" />
              </div>
            </div>
          </SpotlightCard>

          <!-- Batch Generation -->
          <SpotlightCard v-if="generationMode === 'batch'">
            <div class="flex items-center justify-between mb-4">
              <h4 class="text-lg font-medium text-white">Batch Generation</h4>
              <button @click="addBatchSet"
                class="text-purple-400 hover:text-purple-300 flex items-center gap-1 text-sm">
                <UIcon name="i-lucide-plus" class="size-4" />
                Add Set
              </button>
            </div>

            <div class="space-y-6">
              <div v-for="(set, setIndex) in batchGenVars" :key="setIndex"
                class="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <div class="flex items-center justify-between mb-4">
                  <h5 class="text-white font-medium">Set {{ setIndex + 1 }}</h5>
                  <button v-if="batchGenVars.length > 1" @click="removeBatchSet(setIndex)"
                    class="p-1 text-red-400 hover:bg-red-400/10 rounded">
                    <UIcon name="i-lucide-trash-2" class="size-4" />
                  </button>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div v-for="variable in selectedTemplate.variables" :key="variable.name">
                    <label class="block text-sm font-medium text-gray-300 mb-1">
                      {{ variable.name.replace(/_/g, ' ') }}
                      <span v-if="variable.required" class="text-red-400">*</span>
                    </label>
                    <UTextarea v-if="variable.type === 'textarea'" v-model="set[variable.name]"
                      :placeholder="variable.description" rows="2" class="w-full" />
                    <UInput v-else v-model="set[variable.name]" :type="variable.type === 'number' ? 'number' : 'text'"
                      :placeholder="variable.description" class="w-full" />
                  </div>
                </div>
              </div>
            </div>
          </SpotlightCard>

          <!-- CSV Generation -->
          <SpotlightCard v-if="generationMode === 'csv'">
            <h4 class="text-lg font-medium text-white mb-4">CSV Import</h4>

            <div v-if="csvData.length === 0" class="space-y-4">
              <p class="text-muted">Upload a CSV file with columns matching your template variables:</p>

              <!-- Required columns -->
              <div class="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <h5 class="text-white font-medium mb-2">Required CSV Columns:</h5>
                <div class="flex flex-wrap gap-2">
                  <span v-for="variable in selectedTemplate.variables" :key="variable.name"
                    class="px-3 py-1 text-sm rounded"
                    :class="variable.required ? 'bg-red-500/20 text-red-300' : 'bg-gray-500/20 text-gray-300'">
                    {{ variable.name }}{{ variable.required ? ' *' : '' }}
                  </span>
                </div>
              </div>

              <!-- File upload -->
              <div
                class="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-gray-500 transition-colors">
                <input type="file" accept=".csv" @change="handleCSVUpload"
                  class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                <UIcon name="i-lucide-upload" class="size-12 text-gray-400 mx-auto mb-4" />
                <h5 class="text-white font-medium mb-2">Upload CSV File</h5>
                <p class="text-muted">Click to select a CSV file with your variable data</p>
              </div>
            </div>

            <!-- CSV Preview -->
            <div v-else class="space-y-4">
              <div class="flex items-center justify-between">
                <div>
                  <h5 class="text-white font-medium">CSV Data Loaded</h5>
                  <p class="text-muted text-sm">{{ csvData.length }} rows ready for generation</p>
                </div>
                <button @click="csvData = []; csvFile = null"
                  class="text-red-400 hover:text-red-300 flex items-center gap-1 text-sm">
                  <UIcon name="i-lucide-x" class="size-4" />
                  Clear CSV
                </button>
              </div>

              <!-- CSV preview table -->
              <div class="overflow-x-auto">
                <table class="w-full text-sm">
                  <thead>
                    <tr class="border-b border-gray-700">
                      <th class="text-left py-2 px-3 text-gray-300 font-medium">#</th>
                      <th v-for="variable in selectedTemplate.variables" :key="variable.name"
                        class="text-left py-2 px-3 text-gray-300 font-medium">
                        {{ variable.name }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, index) in csvData.slice(0, 5)" :key="index" class="border-b border-gray-800">
                      <td class="py-2 px-3 text-gray-400">{{ index + 1 }}</td>
                      <td v-for="variable in selectedTemplate.variables" :key="variable.name"
                        class="py-2 px-3 text-gray-300">
                        {{ row[variable.name] || '-' }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p v-if="csvData.length > 5" class="text-xs text-gray-400 text-center">
                Showing first 5 rows of {{ csvData.length }} total rows
              </p>
            </div>
          </SpotlightCard>

          <!-- Generation Button and Progress -->
          <SpotlightCard>
            <div v-if="!isGenerating">
              <SpotlightButton @click="startGeneration" :animate="false" class="w-full py-3">
                <div class="flex items-center justify-center gap-2">
                  <UIcon name="i-lucide-play" class="size-5" />
                  <span class="font-medium">
                    Start Generation
                    <span v-if="generationMode === 'single'">(1 piece)</span>
                    <span v-else-if="generationMode === 'batch'">({{ batchGenVars.length }} pieces)</span>
                    <span v-else-if="generationMode === 'csv' && csvData.length > 0">({{ csvData.length }}
                      pieces)</span>
                  </span>
                </div>
              </SpotlightButton>
            </div>

            <!-- Generation Progress -->
            <div v-if="isGenerating" class="space-y-4">
              <div class="flex items-center justify-between">
                <h4 class="text-lg font-medium text-white">Generating Content...</h4>
                <span class="text-muted">{{ generationProgress.current }} / {{ generationProgress.total }}</span>
              </div>

              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                  :style="{ width: `${(generationProgress.current / generationProgress.total) * 100}%` }"></div>
              </div>

              <div class="text-center">
                <div
                  class="animate-spin w-8 h-8 border-4 border-purple-400/20 border-t-purple-400 rounded-full mx-auto mb-2">
                </div>
                <p class="text-muted">Processing your content...</p>
              </div>
            </div>

            <!-- Generation Results -->
            <div v-if="generationProgress.results.length > 0 && !isGenerating" class="space-y-4">
              <div class="flex items-center justify-between">
                <h4 class="text-lg font-medium text-white">Generation Results</h4>
                <button @click="downloadResults"
                  class="text-purple-400 hover:text-purple-300 flex items-center gap-2 text-sm">
                  <UIcon name="i-lucide-download" class="size-4" />
                  Download CSV
                </button>
              </div>

              <div class="space-y-3 max-h-96 overflow-y-auto">
                <div v-for="(result, index) in generationProgress.results" :key="result.id"
                  class="p-4 rounded-lg border"
                  :class="result.status === 'success' ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'">
                  <div class="flex items-start justify-between mb-2">
                    <div class="flex items-center gap-2">
                      <UIcon :name="result.status === 'success' ? 'i-lucide-check' : 'i-lucide-x'" class="size-4"
                        :class="result.status === 'success' ? 'text-green-400' : 'text-red-400'" />
                      <span class="text-white font-medium">Result {{ index + 1 }}</span>
                    </div>
                    <button v-if="result.status === 'success'" @click="copyToClipboard(result.content)"
                      class="p-1 text-gray-400 hover:text-white rounded" title="Copy to clipboard">
                      <UIcon name="i-lucide-copy" class="size-4" />
                    </button>
                  </div>

                  <div v-if="result.status === 'success'" class="text-gray-300 text-sm leading-relaxed">
                    {{ result.content }}
                  </div>
                  <div v-else class="text-red-300 text-sm">
                    Error: {{ result.error }}
                  </div>
                </div>
              </div>
            </div>
          </SpotlightCard>
        </div>
      </div>

      <!-- History Tab -->
      <div v-if="currentTab === 'history'" class="space-y-6">
        <SpotlightCard>
          <div class="flex items-center justify-between mb-6">
            <h3 class="text-xl font-semibold text-white flex items-center gap-2">
              <UIcon name="i-lucide-history" class="size-5 text-cyan-400" />
              Generation History
            </h3>
            <button v-if="generationHistory.length > 0" @click="clearHistory"
              class="text-red-400 hover:text-red-300 flex items-center gap-2 text-sm">
              <UIcon name="i-lucide-trash-2" class="size-4" />
              Clear All
            </button>
          </div>

          <div v-if="generationHistory.length === 0" class="text-center py-12">
            <div class="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <UIcon name="i-lucide-history" class="size-8 text-gray-400" />
            </div>
            <h4 class="text-lg font-medium text-white mb-2">No History Yet</h4>
            <p class="text-muted">Generated content will appear here</p>
          </div>

          <div v-else class="space-y-4 max-h-[600px] overflow-y-auto">
            <div v-for="item in generationHistory" :key="item.id" class="p-4 rounded-lg border"
              :class="item.status === 'success' ? 'border-gray-700 bg-gray-800/50' : 'border-red-500/30 bg-red-500/5'">
              <div class="flex items-start justify-between mb-3">
                <div>
                  <div class="flex items-center gap-2 mb-1">
                    <UIcon :name="item.status === 'success' ? 'i-lucide-check' : 'i-lucide-x'" class="size-4"
                      :class="item.status === 'success' ? 'text-green-400' : 'text-red-400'" />
                    <span class="text-white font-medium">{{ item.templateName }}</span>
                  </div>
                  <div class="text-xs text-gray-400">
                    {{ new Date(item.createdAt).toLocaleString() }}
                  </div>
                </div>
                <button v-if="item.status === 'success'" @click="copyToClipboard(item.content)"
                  class="p-1 text-gray-400 hover:text-white rounded" title="Copy to clipboard">
                  <UIcon name="i-lucide-copy" class="size-4" />
                </button>
              </div>

              <!-- Variables used -->
              <div class="mb-3">
                <div class="flex flex-wrap gap-2">
                  <span v-for="(value, key) in item.variables" :key="key"
                    class="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded" :title="`${key}: ${value}`">
                    {{ key }}: {{ value.toString().substring(0, 20) }}{{ value.toString().length > 20 ? '...' : '' }}
                  </span>
                </div>
              </div>

              <!-- Content -->
              <div v-if="item.status === 'success'"
                class="text-gray-300 text-sm leading-relaxed p-3 bg-gray-900/50 rounded border border-gray-700">
                {{ item.content }}
              </div>
              <div v-else class="text-red-300 text-sm p-3 bg-red-500/10 rounded border border-red-500/30">
                Error: {{ item.error }}
              </div>
            </div>
          </div>
        </SpotlightCard>
      </div>
    </div>
  </div>
</template>