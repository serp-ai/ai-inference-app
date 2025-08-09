<template>
  <div class="background-container">
    <div class="ColorBlobs_SpinningGradient"></div>
    <div class="HeroSection_pulseAnimation"></div>
  </div>
</template>

<style scoped>
.background-container {
  position: fixed;
  /* Ensures it covers the entire viewport and stays fixed */
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  z-index: -1;
  /* Ensures it is in the background */
  display: flex;
  justify-content: center;
  align-items: center;
}

.HeroSection_pulseAnimation {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  width: 80vmin;
  /* Scales with the smallest viewport dimension */
  height: 80vmin;
  position: absolute;
  animation: HeroSection_pulse 4s ease-in-out infinite;
  filter: blur(100px);
}

.ColorBlobs_SpinningGradient {
  animation: ColorBlobs_spin 12s linear infinite;
  background: conic-gradient(from 0deg, #08f, #f60, #bbffa1, #4c00ff, #ab2666, #09f);
  border-radius: 50%;
  width: 80vmin;
  /* Scales with the smallest viewport dimension */
  height: 80vmin;
  position: absolute;
  z-index: -1;
  filter: blur(100px);
  opacity: 0.1;
}

@keyframes HeroSection_pulse {
  0% {
    opacity: 0.5;
    transform: scale(0.9);
  }

  50% {
    opacity: 1;
    transform: scale(1);
  }

  100% {
    opacity: 0.5;
    transform: scale(0.9);
  }
}

@keyframes ColorBlobs_spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
</style>
