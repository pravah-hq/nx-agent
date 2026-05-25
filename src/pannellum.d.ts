type PannellumViewer = {
  destroy?: () => void;
  getHfov?: () => number;
  getPitch?: () => number;
  getYaw?: () => number;
  setPitch?: (pitch: number, animated?: boolean) => void;
  setYaw?: (yaw: number, animated?: boolean) => void;
};

type PannellumApi = {
  viewer: (
    element: HTMLElement,
    options: {
      type: "equirectangular";
      panorama: string;
      autoLoad: boolean;
      yaw?: number;
      pitch?: number;
      hfov?: number;
      showZoomCtrl?: boolean;
    },
  ) => PannellumViewer;
};

interface Window {
  pannellum?: PannellumApi;
}
