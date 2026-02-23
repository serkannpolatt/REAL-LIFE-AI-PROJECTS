# 🎨 AI Stock Analysis Assistant - Frontend

The React frontend for the AI Stock Analysis Assistant, built with TypeScript and Vite.

## 📋 Overview

This frontend provides:
- Modern chat interface for stock analysis
- Real-time streaming responses
- Dark theme for comfortable viewing
- Responsive design for all devices

## 🏗 Architecture

```
src/
├── main.tsx          # Application entry point
├── App.tsx           # Main application component
├── App.css           # Global styles
└── assets/           # Static assets (images, icons)
```

## 📦 Dependencies

### Production Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| react | ^19.2.0 | UI framework |
| react-dom | ^19.2.0 | React DOM rendering |
| @thesysai/genui-sdk | ^0.7.4 | Chat component SDK |
| @crayonai/react-ui | ^0.9.5 | UI component styles |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| typescript | ~5.9.3 | Type checking |
| vite | ^7.2.4 | Build tool |
| @vitejs/plugin-react | ^5.1.1 | React support for Vite |
| eslint | ^9.39.1 | Code linting |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### 3. Build for Production

```bash
npm run build
```

## 📜 Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server with HMR |
| `npm run build` | Build for production |
| `npm run lint` | Run ESLint |
| `npm run preview` | Preview production build |

## 🎯 Components

### App Component

The main application component that renders the chat interface.

```tsx
import { C1Chat, ThemeProvider } from "@thesysai/genui-sdk";

function App() {
  return (
    <ThemeProvider mode="dark">
      <C1Chat apiUrl="/api/chat" />
    </ThemeProvider>
  );
}
```

**Features:**
- Dark theme for reduced eye strain
- Full-screen chat interface
- Automatic API connection

## ⚙ Configuration

### Vite Configuration

The `vite.config.ts` file configures:

1. **Development Server**: Runs on port 3000
2. **API Proxy**: Forwards `/api/*` requests to the backend at `localhost:8888`

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8888",
        changeOrigin: true
      }
    }
  }
});
```

### TypeScript Configuration

Multiple TypeScript configurations:

- `tsconfig.json` - Base configuration
- `tsconfig.app.json` - Application code settings
- `tsconfig.node.json` - Node.js environment settings

## 🎨 Styling

### Global Styles (App.css)

- Reset default browser styles
- Full-viewport layout
- Inter font family throughout

### Theme

The application uses a dark theme provided by TheSys GenUI SDK:

```tsx
<ThemeProvider mode="dark">
  {/* components */}
</ThemeProvider>
```

## 📁 File Structure

```
frontend/
├── public/              # Static files
│   └── vite.svg        # Vite logo
├── src/
│   ├── assets/         # Application assets
│   ├── App.tsx         # Main component
│   ├── App.css         # Global styles
│   └── main.tsx        # Entry point
├── index.html          # HTML template
├── package.json        # Dependencies
├── vite.config.ts      # Vite configuration
├── tsconfig.json       # TypeScript config
└── eslint.config.js    # ESLint configuration
```

## 🔧 Development

### Hot Module Replacement (HMR)

Vite provides instant updates without full page reload:

1. Edit any `.tsx` or `.css` file
2. Changes appear immediately in the browser

### Type Checking

Run TypeScript compiler:

```bash
npx tsc --noEmit
```

### Linting

```bash
npm run lint
```

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend is running on port 8888
   - Check browser console for CORS errors

2. **Styles Not Loading**
   - Verify `@crayonai/react-ui/styles/index.css` is imported

3. **Build Errors**
   - Run `npm run lint` to check for code issues
   - Ensure all dependencies are installed


