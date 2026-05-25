import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const manifestPath = path.join(root, "data", "metadata", "panoramas.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

const missing = [];
for (const pano of manifest.panoramas) {
  const imagePath = path.join(root, "data", "panoramas", pano.imagePath);
  if (!fs.existsSync(imagePath)) missing.push(pano.imagePath);
}

if (missing.length) {
  console.error(`Missing ${missing.length} of ${manifest.panoramas.length} panorama images.`);
  for (const item of missing.slice(0, 20)) console.error(`- ${item}`);
  if (missing.length > 20) console.error(`- ...and ${missing.length - 20} more`);
  process.exit(1);
}

console.log(`Found all ${manifest.panoramas.length} panorama images.`);
