/**
 * AI Stock Analysis Assistant - Main Application Component
 *
 * This component serves as the root of the React application,
 * integrating the TheSys GenUI SDK for the chat interface
 * with a dark theme design.
 *
 * @author AI Stock Analysis Team
 * @version 1.0.0
 */

// ==============================================================================
// IMPORTS
// ==============================================================================

// TheSys GenUI SDK Components
// C1Chat: Pre-built chat component that connects to the backend API
// ThemeProvider: Provides theming support (dark/light mode)
import { C1Chat, ThemeProvider } from "@thesysai/genui-sdk";

// Crayon AI UI Styles - Required base styles for the chat component
import "@crayonai/react-ui/styles/index.css";

// Custom Application Styles
import "./App.css";

// ==============================================================================
// CONSTANTS
// ==============================================================================

/**
 * API endpoint URL for the chat backend
 * This is proxied through Vite dev server to the FastAPI backend
 */
const API_CHAT_URL = "/api/chat";

/**
 * Theme mode for the application
 * Options: 'dark' | 'light'
 */
const THEME_MODE = "dark";

// ==============================================================================
// MAIN APPLICATION COMPONENT
// ==============================================================================

/**
 * App Component
 *
 * The main application component that renders the stock analysis chat interface.
 * It wraps the C1Chat component with ThemeProvider for consistent styling.
 *
 * Features:
 * - Dark theme for better readability
 * - Full-screen chat interface
 * - Real-time streaming responses from the AI backend
 *
 * @returns {JSX.Element} The rendered application
 */
function App(): JSX.Element {
  return (
    <div className="app-container">
      {/* ThemeProvider wraps the entire chat component for consistent theming */}
      <ThemeProvider mode={THEME_MODE}>
        {/* C1Chat component handles all chat functionality including:
            - Message input and display
            - Real-time streaming of AI responses
            - Conversation history management
        */}
        <C1Chat apiUrl={API_CHAT_URL} />
      </ThemeProvider>
    </div>
  );
}

// ==============================================================================
// EXPORTS
// ==============================================================================

export default App;
