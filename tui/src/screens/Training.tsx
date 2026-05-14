import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';

import type { BackendState } from '../hooks/useBackendSession';

export function Training({
  session,
}: {
  session: BackendState & { sendRequest: (req: any) => void };
}): React.JSX.Element {
  const [modelType, setModelType] = useState('lm');
  const [config, setConfig] = useState('');
  const [configMode, setConfigMode] = useState(false);

  const handleLaunch = (): void => {
    const cmd = ['python', '-m', `src.tui.bin.train_${modelType}`, '--config', config].filter(
      Boolean,
    );
    session.sendRequest({
      type: 'train_launch',
      model_type: modelType as any,
      train_cmd: cmd,
    });
  };

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold underline>
        Training — Launch & Monitor
      </Text>
      <Box marginY={0} />

      <Box flexDirection="row" marginY={0} columnGap={2}>
        <Box flexDirection="row">
          <Text bold>Model: </Text>
          <TextInput
            value={modelType}
            onChange={setModelType}
            placeholder="lm | vlm | rlm"
          />
        </Box>
      </Box>

      <Box flexDirection="row" marginY={0}>
        <Text bold>Config: </Text>
        <TextInput
          value={config}
          onChange={setConfig}
          placeholder="Path to training config..."
        />
      </Box>

      <Box marginY={0}>
        <Text color="blue" bold>
          {'> '}Press Enter to launch training
        </Text>
      </Box>

      {/* Status */}
      <Box
        flexDirection="column"
        flexGrow={1}
        borderStyle="round"
        borderColor="gray"
        paddingX={1}
        marginY={0}
      >
        <Text bold>Training Log</Text>
        <Box marginY={0} />
        {session.trainStatus ? (
          <Box flexDirection="column">
            {Object.entries(session.trainStatus).map(([k, v]) => (
              <Text key={k}>
                {k}: {String(v ?? '')}
              </Text>
            ))}
          </Box>
        ) : (
          <Text dimColor>No training jobs running. Launch one above.</Text>
        )}
      </Box>

      <Box>
        <Text dimColor>
          Configure model type and config path, then press Enter to launch.
        </Text>
      </Box>
    </Box>
  );
}
