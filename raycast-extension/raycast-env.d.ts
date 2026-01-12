/// <reference types="@raycast/api">

/* 🚧 🚧 🚧
 * This file is auto-generated from the extension's manifest.
 * Do not modify manually. Instead, update the `package.json` file.
 * 🚧 🚧 🚧 */

/* eslint-disable @typescript-eslint/ban-types */

type ExtensionPreferences = {
  /** Proxy URL - The base URL of your LLM proxy (e.g., http://localhost:8000) */
  "proxyUrl": string,
  /** API Key - Your proxy API key (PROXY_API_KEY). Leave empty if proxy is unsecured. */
  "apiKey"?: string
}

/** Preferences accessible in all the extension's commands */
declare type Preferences = ExtensionPreferences

declare namespace Preferences {
  /** Preferences accessible in the `quota` command */
  export type Quota = ExtensionPreferences & {}
}

declare namespace Arguments {
  /** Arguments passed to the `quota` command */
  export type Quota = {}
}

