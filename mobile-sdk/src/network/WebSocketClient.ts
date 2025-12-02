/**
 * WebSocket Client
 * Manages WebSocket connection to orchestrator
 */

import { EventEmitter } from 'events';
import { Task } from '../types';

export class WebSocketClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectDelay: number = 1000;
  private isConnected: boolean = false;

  constructor(private url: string) {
    super();
  }

  connect(): void {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      };

      this.ws.onerror = (error) => {
        this.emit('error', error);
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this.emit('disconnected');
        this.attemptReconnect();
      };
    } catch (error) {
      this.emit('error', error);
      this.attemptReconnect();
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message: any): void {
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify(message));
    } else {
      throw new Error('WebSocket not connected');
    }
  }

  private handleMessage(message: any): void {
    switch (message.type) {
      case 'task':
        this.emit('task', message.data as Task);
        break;
      case 'ping':
        this.send({ type: 'pong' });
        break;
      default:
        this.emit('message', message);
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      this.emit('reconnectFailed');
    }
  }
}
