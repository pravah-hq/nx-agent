import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dataRoot = path.join(root, "data");
const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 8787);

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".geojson": "application/geo+json; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function send(res, status, body, headers = {}) {
  res.writeHead(status, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    ...headers,
  });
  res.end(body);
}

function safeDataPath(urlPath) {
  let relativePath;
  if (urlPath.startsWith("/metadata/")) {
    relativePath = urlPath.slice(1);
  } else if (urlPath.startsWith("/data/")) {
    relativePath = urlPath.slice("/data/".length);
  } else {
    return null;
  }
  const decoded = decodeURIComponent(relativePath);
  const target = path.normalize(path.join(dataRoot, decoded));
  if (!target.startsWith(dataRoot + path.sep)) return null;
  return target;
}

const server = http.createServer((req, res) => {
  if (!req.url) return send(res, 400, "bad request\n");
  if (req.method === "OPTIONS") return send(res, 204, "");
  if (req.method !== "GET" && req.method !== "HEAD") return send(res, 405, "method not allowed\n");

  const url = new URL(req.url, `http://${host}:${port}`);
  if (url.pathname === "/health") {
    return send(res, 200, req.method === "HEAD" ? "" : '{"ok":true}\n', {
      "Content-Type": "application/json; charset=utf-8",
    });
  }

  const target = safeDataPath(url.pathname);
  if (!target) return send(res, 404, "not found\n");

  fs.stat(target, (statErr, stat) => {
    if (statErr || !stat.isFile()) return send(res, 404, "not found\n");
    const contentType = contentTypes[path.extname(target).toLowerCase()] || "application/octet-stream";
    res.writeHead(200, {
      "Access-Control-Allow-Origin": "*",
      "Content-Type": contentType,
      "Content-Length": stat.size,
    });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(target).pipe(res);
  });
});

server.listen(port, host, () => {
  console.log(`backend listening on http://${host}:${port}`);
});
