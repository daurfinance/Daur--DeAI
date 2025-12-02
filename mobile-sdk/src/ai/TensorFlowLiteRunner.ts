/**
 * TensorFlow Lite Runner
 * Runs TensorFlow Lite models on mobile devices
 */

import { NativeModules } from 'react-native';

const { TFLiteModule } = NativeModules;

export class TensorFlowLiteRunner {
  private modelLoaded: boolean = false;

  async loadModel(modelPath: string): Promise<void> {
    try {
      await TFLiteModule.loadModel(modelPath);
      this.modelLoaded = true;
    } catch (error) {
      throw new Error(`Failed to load model: ${error}`);
    }
  }

  async runInference(input: any): Promise<any> {
    if (!this.modelLoaded) {
      throw new Error('No model loaded');
    }

    try {
      const result = await TFLiteModule.runInference(input);
      return result;
    } catch (error) {
      throw new Error(`Inference failed: ${error}`);
    }
  }

  async train(trainingData: any[], epochs: number): Promise<any> {
    if (!this.modelLoaded) {
      throw new Error('No model loaded');
    }

    try {
      // Run federated learning training
      const result = await TFLiteModule.train({
        data: trainingData,
        epochs,
      });
      
      return {
        weights: result.weights,
        loss: result.loss,
        accuracy: result.accuracy,
      };
    } catch (error) {
      throw new Error(`Training failed: ${error}`);
    }
  }

  async unloadModel(): Promise<void> {
    if (this.modelLoaded) {
      await TFLiteModule.unloadModel();
      this.modelLoaded = false;
    }
  }
}
