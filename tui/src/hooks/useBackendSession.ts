import { useEffect, useMemo, useRef, useState } from 'react';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import readline from 'node:readline';

import type {
  BackendEvent,
  DatasetEntry,
  DatasetStats,
  FrontendConfig,
  FrontendRequest,
  ModelSnapshot,
  SampleResult,
} from '../types';

const PROTOCOL_PREFIX = 'VMJSON:';

export type BackendState = {
  ready: boolean;
  modelList: ModelSnapshot[];
  modelStatus: string;
  datasets: DatasetEntry[];
  datasetStats: DatasetStats | null;
  datasetSample: SampleResult | null;
  tokens: string[];
  generationDone: boolean;
  error: string | null;
  trainStatus: Record<string, unknown> | null;
  benchResult: Record<string, unknown> | null;
  backendEvent: BackendEvent | null;
};

const INITIAL_STATE: BackendState = {
  ready: false,
  modelList: [],
  modelStatus: '',
  datasets: [],
  datasetStats: null,
  datasetSample: null,
  tokens: [],
  generationDone: false,
  error: null,
  trainStatus: null,
  benchResult: null,
  backendEvent: null,
};

export function useBackendSession(
  config: FrontendConfig,
  onExit: (code?: number | null) => void,
) {
  const [state, setState] = useState<BackendState>(INITIAL_STATE);
  const childRef = useRef<ChildProcessWithoutNullStreams | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  const sendRequest = (payload: FrontendRequest): void => {
    const child = childRef.current;
    if (!child || child.stdin.destroyed) {
      return;
    }
    child.stdin.write(JSON.stringify(payload) + '\n');
  };

  useEffect(() => {
    const [command, ...args] = config.backend_command;
    const child = spawn(command, args, {
      stdio: ['pipe', 'pipe', 'inherit'],
      env: process.env,
      cwd: config.cwd ?? process.cwd(),
    });
    childRef.current = child;

    const reader = readline.createInterface({ input: child.stdout });
    reader.on('line', (line: string) => {
      if (!line.startsWith(PROTOCOL_PREFIX)) {
        return;
      }
      try {
        const event = JSON.parse(line.slice(PROTOCOL_PREFIX.length)) as BackendEvent;
        handleEvent(event);
      } catch {
        // ignore malformed lines
      }
    });

    child.on('exit', (code) => {
      onExit(code);
    });

    const killChild = (): void => {
      if (child && !child.killed) {
        child.kill('SIGTERM');
      }
    };
    process.on('exit', killChild);

    return () => {
      reader.close();
      killChild();
      process.removeListener('exit', killChild);
    };
  }, []);

  const handleEvent = (event: BackendEvent): void => {
    switch (event.type) {
      case 'ready':
        setState((s) => ({ ...s, ready: true }));
        break;
      case 'model_list':
        setState((s) => ({ ...s, modelList: event.data ?? [] }));
        break;
      case 'model_loaded':
        setState((s) => ({ ...s, modelStatus: event.message }));
        break;
      case 'token':
        setState((s) => ({
          ...s,
          tokens: [...s.tokens, event.message],
        }));
        break;
      case 'done':
        setState((s) => ({ ...s, generationDone: true }));
        break;
      case 'error':
        setState((s) => ({ ...s, error: event.message }));
        break;
      case 'metrics':
        break;
      case 'eda_result':
        setState((s) => ({ ...s, datasets: (event.data ?? []) as DatasetEntry[] }));
        break;
      case 'eda_stats_result':
        setState((s) => ({ ...s, datasetStats: (event.data ?? null) as DatasetStats | null }));
        break;
      case 'eda_sample_result':
        setState((s) => ({ ...s, datasetSample: (event.data ?? null) as SampleResult | null }));
        break;
      case 'bench_result':
        setState((s) => ({ ...s, benchResult: event.data ?? null }));
        break;
      case 'train_log':
        break;
      case 'train_status':
        setState((s) => ({ ...s, trainStatus: event.data ?? null }));
        break;
      case 'shutdown':
        setState((s) => ({ ...s, ready: false }));
        onExit(0);
        break;
    }
  };

  return useMemo(
    () => ({
      ...state,
      sendRequest,
      clearGeneration: () =>
        setState((s) => ({ ...s, tokens: [], generationDone: false, error: null })),
      clearEda: () =>
        setState((s) => ({ ...s, datasets: [], datasetStats: null, datasetSample: null })),
    }),
    [
      state.ready,
      state.modelList,
      state.modelStatus,
      state.datasets,
      state.datasetStats,
      state.datasetSample,
      state.tokens,
      state.generationDone,
      state.error,
      state.trainStatus,
      state.benchResult,
    ],
  );
}
