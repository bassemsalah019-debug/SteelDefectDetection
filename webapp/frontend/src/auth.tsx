import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setTokenGetter, type Token, type User } from "./api";

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, fullName: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState>(null as unknown as AuthState);
export const useAuth = () => useContext(AuthContext);

const TOKEN_KEY = "sv_token";
const REFRESH_KEY = "sv_refresh";

setTokenGetter(() => localStorage.getItem(TOKEN_KEY));

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      if (localStorage.getItem(TOKEN_KEY)) {
        try { const u = await api.me(); if (active) setUser(u); }
        catch { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REFRESH_KEY); }
      }
      if (active) setLoading(false);
    })();
    return () => { active = false; };
  }, []);

  const persist = (t: Token) => {
    localStorage.setItem(TOKEN_KEY, t.access_token);
    localStorage.setItem(REFRESH_KEY, t.refresh_token);
    setUser(t.user);
  };

  const login = async (email: string, password: string) => persist(await api.login({ email, password }));
  const signup = async (email: string, password: string, fullName: string) =>
    persist(await api.signup({ email, password, full_name: fullName }));
  const logout = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
