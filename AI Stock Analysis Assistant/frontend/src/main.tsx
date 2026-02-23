/**
 * AI Stock Analysis Assistant - Application Entry Point
 *
 * This file is the main entry point for the React application.
 * It initializes the React root and renders the App component
 * with StrictMode enabled for development checks.
 *
 * @author AI Stock Analysis Team
 * @version 1.0.0
 */

// ==============================================================================
// IMPORTS
// ==============================================================================

// React Core
import { StrictMode } from "react";

// React DOM - Client-side rendering utilities
import { createRoot } from "react-dom/client";

// Main Application Component
import App from "./App.tsx";

// ==============================================================================
// APPLICATION INITIALIZATION
// ==============================================================================

/**
 * Get the root DOM element where the React app will be mounted.
 * The '!' is a non-null assertion since we know the element exists in index.html
 */
const rootElement = document.getElementById("root")!;

/**
 * Create the React root and render the application.
 *
 * StrictMode is enabled to help identify potential problems in the application:
 * - Identifies components with unsafe lifecycles
 * - Warns about legacy string ref API usage
 * - Warns about deprecated findDOMNode usage
 * - Detects unexpected side effects
 * - Ensures reusable state
 *
 * Note: StrictMode only runs in development mode and has no impact on production.
 */
createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
