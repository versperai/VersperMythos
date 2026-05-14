import React from 'react';
import { render } from 'ink';
import fs from 'node:fs';
import tty from 'node:tty';

import { App } from './App';
import type { FrontendConfig } from './types';

// Guard against EIO crashes in terminal environments
process.stdin.on('error', (err: NodeJS.ErrnoException) => {
  if (err.code === 'EIO' || err.code === 'EAGAIN') {
    process.exit(1);
  }
  throw err;
});

if (process.stdin.isTTY && typeof process.stdin.setRawMode === 'function') {
  const origSetRawMode = process.stdin.setRawMode.bind(process.stdin);
  process.stdin.setRawMode = (mode: boolean) => {
    try {
      return origSetRawMode(mode);
    } catch (err: any) {
      if (err?.code === 'EIO' || err?.code === 'EAGAIN') {
        process.exit(1);
      }
      throw err;
    }
  };
}

process.on('uncaughtException', (err: NodeJS.ErrnoException) => {
  if (err.code === 'EIO' || err.code === 'EAGAIN') {
    process.exit(1);
  }
  throw err;
});

// Restore cursor on exit (Ink hides it)
const restoreCursor = (): void => {
  process.stdout.write('\x1B[?25h');
};
process.on('exit', restoreCursor);
process.on('SIGINT', () => {
  restoreCursor();
  process.exit(130);
});
process.on('SIGTERM', () => {
  restoreCursor();
  process.exit(143);
});

const config = JSON.parse(process.env.VM_TUI_CONFIG ?? '{}') as FrontendConfig;

// TTY fallback
let stdinStream: NodeJS.ReadStream & { fd: 0 } = process.stdin;
let ttyFd: number | undefined;

if (!process.stdin.isTTY) {
  try {
    ttyFd = fs.openSync('/dev/tty', 'r');
    const ttyStream = new tty.ReadStream(ttyFd);
    stdinStream = ttyStream as unknown as NodeJS.ReadStream & { fd: 0 };
  } catch {
    // non-interactive fallback
  }
}

process.on('exit', () => {
  if (ttyFd !== undefined) {
    try {
      fs.closeSync(ttyFd);
    } catch {
      /* ignore */
    }
  }
});

render(<App config={config} />, { stdin: stdinStream });
