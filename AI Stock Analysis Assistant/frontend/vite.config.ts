/**
 * Vite Configuration for AI Stock Analysis Assistant
 *
 * This configuration file sets up the Vite development server
 * with React support and API proxy settings for connecting
 * to the FastAPI backend.
 *
 * @see https://vite.dev/config/
 * @author AI Stock Analysis Team
 * @version 1.0.0
 */

// ==============================================================================
// IMPORTS
// ==============================================================================

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// ==============================================================================
// CONFIGURATION CONSTANTS
// ==============================================================================

/**
 * Development server port
 * The frontend will be accessible at http://localhost:3000
 */
const DEV_SERVER_PORT = 3000;

/**
 * Backend API server URL
 * The FastAPI backend runs on this address
 */
const BACKEND_API_URL = "http://localhost:8888";

// ==============================================================================
// VITE CONFIGURATION
// ==============================================================================

export default defineConfig({
  /**
   * Plugins Configuration
   * - react(): Enables React Fast Refresh and JSX support
   */
  plugins: [react()],

  /**
   * Development Server Configuration
   */
  server: {
    /**
     * Port number for the dev server
     * Access the app at http://localhost:3000
     */
    port: DEV_SERVER_PORT,

    /**
     * Proxy Configuration
     *
     * Routes all /api/* requests to the FastAPI backend.
     * This enables seamless communication between frontend and backend
     * during development without CORS issues.
     *
     * Example:
     * - Frontend request: POST /api/chat
     * - Proxied to: POST http://localhost:8888/api/chat
     */
    proxy: {
      "/api": {
        target: BACKEND_API_URL,
        changeOrigin: true, // Changes the origin of the request to the target URL
      },
    },
  },
});
