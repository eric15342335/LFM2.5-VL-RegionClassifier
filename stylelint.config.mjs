export default {
  extends: ["stylelint-config-standard"],
  rules: {
    "no-descending-specificity": [
      true,
      {
        severity: "warning",
      },
    ],
  },
  reportDescriptionlessDisables: true,
  reportNeedlessDisables: true,
  reportUnscopedDisables: true,
};
