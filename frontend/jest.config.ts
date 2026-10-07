import type { Config } from "jest";

/**
 * Phase 13 frontend test configuration.
 *
 * Single framework: Jest + ts-jest + @testing-library/react (jsdom).
 * Vitest is not configured or invoked anywhere.
 */
const config: Config = {
  testEnvironment: "jsdom",
  roots: ["<rootDir>/tests"],
  setupFilesAfterEnv: ["<rootDir>/tests/setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        // ts-jest emits CommonJS for Jest regardless of the app tsconfig,
        // which targets ESM for Next.js.
        tsconfig: {
          jsx: "react-jsx",
          module: "commonjs",
          moduleResolution: "node",
          esModuleInterop: true,
          resolveJsonModule: true,
          allowJs: true,
          skipLibCheck: true,
        },
      },
    ],
  },
  testMatch: ["<rootDir>/tests/**/*.test.ts?(x)"],
  testPathIgnorePatterns: ["/node_modules/", "/\\.next/"],
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json", "node"],
  clearMocks: true,
  restoreMocks: true,
};

export default config;
