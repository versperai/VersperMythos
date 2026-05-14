import React, { useCallback, useMemo, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';

import { Dashboard } from './screens/Dashboard';
import { DataEDA } from './screens/DataEDA';
import { Inference } from './screens/Inference';
import { Training } from './screens/Training';
import { Benchmark } from './screens/Benchmark';
import { SelectModal, type SelectGroup } from './components/SelectModal';
import { useBackendSession } from './hooks/useBackendSession';
import type { FrontendConfig, ModelSnapshot } from './types';

type TabId = 'dashboard' | 'eda' | 'inference' | 'training' | 'benchmark';

const TABS: { id: TabId; label: string; key: string }[] = [
  { id: 'dashboard', label: 'Dashboard', key: '1' },
  { id: 'eda', label: 'Data EDA', key: '2' },
  { id: 'inference', label: 'Inference', key: '3' },
  { id: 'training', label: 'Training', key: '4' },
  { id: 'benchmark', label: 'Benchmark', key: '5' },
];

const TAB_LABELS: Record<TabId, string> = {
  dashboard: 'board',
  eda: 'eda',
  inference: 'infer',
  training: 'train',
  benchmark: 'eval',
};

type UIMode =
  | { type: 'normal' }
  | { type: 'command'; input: string }
  | { type: 'modelPicker' };

export function App({ config }: { config: FrontendConfig }): React.JSX.Element {
  const { exit } = useApp();
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');
  const [mode, setMode] = useState<UIMode>({ type: 'normal' });
  const session = useBackendSession(config, () => exit());

  const handleCommand = useCallback(
    (line: string): boolean => {
      const trimmed = line.trim().toLowerCase();
      if (trimmed === 'model' || trimmed === '/model') {
        // Trigger rediscovery, then show picker
        session.sendRequest({ type: 'discover' });
        setMode({ type: 'modelPicker' });
        return true;
      }
      return false;
    },
    [session.sendRequest],
  );

  // Build select groups from the discovered model list
  const modelPickerGroups: SelectGroup[] = useMemo(() => {
    const list = (session.modelList ?? []) as any[];
    const groups = new Map<string, SelectGroup>();

    for (const m of list) {
      const type: string = m.type ?? 'lm';
      const name: string = m.name ?? '';
      const weightName = name.replace('.pth', '');
      const sizeMB = m.size_mb ?? 0;
      const label = `${weightName}  (${m.hidden_size ?? '?'}dim, ${sizeMB.toFixed(0)}MB)`;

      if (!groups.has(type)) {
        groups.set(type, { title: type.toUpperCase(), items: [] });
      }
      groups.get(type)!.items.push({
        value: JSON.stringify({ type, weight: weightName, path: m.path ?? '' }),
        label,
        description: `${type}`,
      });
    }
    return [...groups.values()];
  }, [session.modelList]);

  useInput((chunk, key) => {
    // modelPicker mode: defer to SelectModal
    if (mode.type === 'modelPicker') return;

    // Command mode: only handle Enter/Escape
    if (mode.type === 'command') {
      if (key.escape) {
        setMode({ type: 'normal' });
        return;
      }
      // TextInput handles characters; Enter submit is via commandInput's onSubmit
      return;
    }

    // Normal mode
    if (key.ctrl && chunk === 'c') {
      session.sendRequest({ type: 'shutdown' });
      exit();
      return;
    }
    if (key.escape) {
      session.sendRequest({ type: 'shutdown' });
      exit();
      return;
    }
    // Shift+Tab: cycle tabs backward
    if (key.tab && key.shift) {
      setActiveTab((tab) => {
        const idx = TABS.findIndex((t) => t.id === tab);
        return TABS[idx <= 0 ? TABS.length - 1 : idx - 1].id;
      });
      return;
    }
    // / key → enter command mode
    if (chunk === '/') {
      setMode({ type: 'command', input: '/' });
      return;
    }
    // Tab navigation with number keys
    if (chunk === '1') setActiveTab('dashboard');
    else if (chunk === '2') setActiveTab('eda');
    else if (chunk === '3') setActiveTab('inference');
    else if (chunk === '4') setActiveTab('training');
    else if (chunk === '5') setActiveTab('benchmark');
  });

  // Show the model picker overlay
  if (mode.type === 'modelPicker') {
    return (
      <SelectModal
        groups={modelPickerGroups}
        title="/model — Select a model"
        onSelect={(value) => {
          try {
            const sel = JSON.parse(value);
            session.sendRequest({ type: 'load_model', model_type: sel.type, weight: sel.weight });
          } catch {
            // fallback
          }
          setMode({ type: 'normal' });
        }}
        onCancel={() => setMode({ type: 'normal' })}
      />
    );
  }

  const renderScreen = (): React.JSX.Element => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard session={session} />;
      case 'eda':
        return <DataEDA session={session} />;
      case 'inference':
        return <Inference session={session} />;
      case 'training':
        return <Training session={session} />;
      case 'benchmark':
        return <Benchmark session={session} />;
    }
  };

  return (
    <Box flexDirection="column">
      {/* Active screen */}
      <Box flexGrow={1} paddingX={1} paddingY={0}>
        {renderScreen()}
      </Box>

      {/* Command input line */}
      {mode.type === 'command' ? (
        <Box flexDirection="row" paddingX={1} paddingY={0}>
          <Text bold color="yellow">{'> '}</Text>
          <TextInput
            value={mode.input}
            onChange={(v) => setMode({ type: 'command', input: v })}
            onSubmit={(v) => {
              if (!handleCommand(v)) {
                // Unknown command — just show it
              }
            }}
            placeholder="type 'model' and press Enter"
          />
        </Box>
      ) : null}

      {/* Status bar */}
      <Box paddingX={1} paddingY={0}>
        <Text dimColor>
          {session.ready ? (
            <Text>
              <Text color="green">●</Text> backend connected
            </Text>
          ) : (
            <Text>
              <Text color="yellow">●</Text> connecting...
            </Text>
          )}
          <Text> │ </Text>
          <Text>
            <Text bold color="cyan">[{TAB_LABELS[activeTab]}]</Text>{' '}
          </Text>
          <Text>│ </Text>
          <Text>
            <Text bold>S-Tab</Text> switch
          </Text>
          <Text> │ </Text>
          <Text>
            <Text bold>/</Text> commands
          </Text>
          <Text> │ </Text>
          <Text>
            <Text bold>Esc</Text> exit
          </Text>
          {session.modelStatus ? (
            <Text>
              <Text> │ </Text>
              <Text color="green">{session.modelStatus}</Text>
            </Text>
          ) : null}
        </Text>
      </Box>
    </Box>
  );
}
