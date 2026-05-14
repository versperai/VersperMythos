import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';

import type { BackendState } from '../hooks/useBackendSession';
import type { DatasetEntry } from '../types';

export function DataEDA({
  session,
}: {
  session: BackendState & { sendRequest: (req: any) => void };
}): React.JSX.Element {
  const [dataDir, setDataDir] = useState('./dataset');
  const [selectedIdx, setSelectedIdx] = useState(0);

  // Scan datasets on mount
  useEffect(() => {
    session.sendRequest({ type: 'eda_scan', data_dir: dataDir });
  }, [dataDir]);

  const datasets = session.datasets as DatasetEntry[];

  // Clear data request
  const selectDataset = (entry: DatasetEntry): void => {
    session.sendRequest({ type: 'eda_stats', file_path: entry.path });
    session.sendRequest({ type: 'eda_sample', file_path: entry.path, index: 0 });
  };

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold underline>
        Data EDA — Dataset Explorer
      </Text>
      <Box marginY={0} />

      <Box flexDirection="row" marginY={0}>
        <Text bold>Data dir: </Text>
        <Text>{dataDir}</Text>
      </Box>

      <Box marginY={0} />

      <Box flexDirection="row" marginY={0}>
        {/* Dataset list */}
        <Box flexDirection="column" width={40} borderStyle="round" borderColor="gray" paddingX={1}>
          <Text bold>Datasets ({datasets.length})</Text>
          <Box marginY={0} />
          {datasets.length === 0 ? (
            <Text dimColor>No datasets found. Scan a directory with datasets.</Text>
          ) : (
            datasets.map((entry, i) => (
              <Box
                key={entry.path}
                flexDirection="row"
                justifyContent="space-between"
              >
                <Box>
                  <Text color={i === selectedIdx ? 'blue' : undefined} bold={i === selectedIdx}>
                    {i === selectedIdx ? '▸ ' : '  '}
                    {entry.name}
                  </Text>
                </Box>
                <Text dimColor>
                  {entry.type.toUpperCase()} {entry.size_mb.toFixed(1)}MB
                </Text>
              </Box>
            ))
          )}
        </Box>

        {/* Stats panel */}
        <Box
          flexDirection="column"
          width={40}
          borderStyle="round"
          borderColor="gray"
          paddingX={1}
        >
          <Text bold>Stats</Text>
          <Box marginY={0} />
          {session.datasetStats ? (
            <Box flexDirection="column">
              {Object.entries(session.datasetStats).map(([k, v]) => (
                <Text key={k}>
                  {k}: {String(v ?? '')}
                </Text>
              ))}
            </Box>
          ) : (
            <Text dimColor>Select a dataset to view stats.</Text>
          )}
        </Box>
      </Box>

      {/* Sample preview */}
      <Box
        flexDirection="column"
        height={10}
        borderStyle="round"
        borderColor="gray"
        paddingX={1}
        marginY={0}
      >
        <Text bold>Sample Preview</Text>
        <Box marginY={0} />
        {session.datasetSample ? (
          <Box flexDirection="column">
            <Text dimColor>{JSON.stringify(session.datasetSample, null, 2).slice(0, 800)}</Text>
          </Box>
        ) : (
          <Text dimColor>Select a dataset to see a sample.</Text>
        )}
      </Box>

      {/* Input hint */}
      <Box>
        <Text dimColor>Use ↑↓ to select datasets, Enter to view stats/sample.</Text>
      </Box>
    </Box>
  );
}
