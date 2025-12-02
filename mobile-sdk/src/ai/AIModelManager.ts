/**
 * AI Model Manager
 * Manages AI models and inference/training
 */

import { TensorFlowLiteRunner } from './TensorFlowLiteRunner';

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  size: number;
  path: string;
}

export class AIModelManager {
  private models: Map<string, ModelInfo> = new Map();
  private tfLiteRunner: TensorFlowLiteRunner;

  constructor() {
    this.tfLiteRunner = new TensorFlowLiteRunner();
  }

  async loadModel(modelInfo: ModelInfo): Promise<void> {
    await this.tfLiteRunner.loadModel(modelInfo.path);
    this.models.set(modelInfo.id, modelInfo);
  }

  async runInference(data: any): Promise<any> {
    const { modelId, input } = data;
    
    if (!this.models.has(modelId)) {
      throw new Error(`Model ${modelId} not loaded`);
    }

    return await this.tfLiteRunner.runInference(input);
  }

  async runTraining(data: any): Promise<any> {
    // Federated learning training
    const { modelId, trainingData, epochs } = data;
    
    if (!this.models.has(modelId)) {
      throw new Error(`Model ${modelId} not loaded`);
    }

    // Run local training
    const result = await this.tfLiteRunner.train(trainingData, epochs || 1);
    
    return {
      modelUpdates: result.weights,
      loss: result.loss,
      accuracy: result.accuracy,
      samplesProcessed: trainingData.length,
    };
  }

  getLoadedModels(): ModelInfo[] {
    return Array.from(this.models.values());
  }

  async unloadModel(modelId: string): Promise<void> {
    this.models.delete(modelId);
    await this.tfLiteRunner.unloadModel();
  }
}
