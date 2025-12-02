/**
 * Device Manager
 * 
 * Manages device capabilities and resource monitoring.
 */

import { DeviceCapabilities, TaskRequirements } from '../types';
import { Platform, NativeModules } from 'react-native';

const { DeviceInfo } = NativeModules;

export class DeviceManager {
  private capabilities: DeviceCapabilities | null = null;
  private monitoringInterval: any = null;

  async initialize(): Promise<void> {
    await this.updateCapabilities();
    this.startMonitoring();
  }

  async getCapabilities(): Promise<DeviceCapabilities> {
    if (!this.capabilities) {
      await this.updateCapabilities();
    }
    return this.capabilities!;
  }

  getCurrentCapabilities(): DeviceCapabilities {
    return this.capabilities!;
  }

  async meetsRequirements(requirements: TaskRequirements): Promise<boolean> {
    const caps = await this.getCapabilities();

    if (requirements.minCPUCores && caps.cpuCores < requirements.minCPUCores) {
      return false;
    }

    if (requirements.minRAM && caps.availableRAM < requirements.minRAM) {
      return false;
    }

    if (requirements.requiresGPU && !caps.gpuModel) {
      return false;
    }

    if (requirements.minBattery && caps.batteryLevel < requirements.minBattery) {
      return false;
    }

    if (requirements.requiresCharging && !caps.isCharging) {
      return false;
    }

    if (requirements.requiresWiFi && caps.networkType !== 'wifi') {
      return false;
    }

    return true;
  }

  private async updateCapabilities(): Promise<void> {
    try {
      // Get device info from native modules
      const deviceInfo = await DeviceInfo.getDeviceInfo();
      
      this.capabilities = {
        cpuCores: deviceInfo.cpuCores || 4,
        cpuFrequency: deviceInfo.cpuFrequency || 2000,
        totalRAM: deviceInfo.totalRAM || 4096,
        availableRAM: deviceInfo.availableRAM || 2048,
        gpuModel: deviceInfo.gpuModel,
        gpuCapability: deviceInfo.gpuCapability,
        batteryLevel: deviceInfo.batteryLevel || 100,
        isCharging: deviceInfo.isCharging || false,
        temperature: deviceInfo.temperature,
        networkType: deviceInfo.networkType || 'wifi',
        networkSpeed: deviceInfo.networkSpeed,
      };
    } catch (error) {
      // Fallback to default values
      this.capabilities = {
        cpuCores: 4,
        cpuFrequency: 2000,
        totalRAM: 4096,
        availableRAM: 2048,
        batteryLevel: 100,
        isCharging: false,
        networkType: 'wifi',
      };
    }
  }

  private startMonitoring(): void {
    // Update capabilities every 30 seconds
    this.monitoringInterval = setInterval(() => {
      this.updateCapabilities();
    }, 30000);
  }

  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }
}
