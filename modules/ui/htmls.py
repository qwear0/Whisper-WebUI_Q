CSS = """
.bmc-button {
    padding: 2px 5px;
    border-radius: 5px;
    background-color: #FF813F;
    color: white;
    box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.3);
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    margin: 2px;
    cursor: pointer;
    -webkit-transition: background-color 0.3s ease;
    -ms-transition: background-color 0.3s ease;
    transition: background-color 0.3s ease;
}
.bmc-button:hover,
.bmc-button:active,
.bmc-button:focus {
    background-color: #FF5633;
}
.markdown {
    margin-bottom: 0;
    padding-bottom: 0;
}
.tabs {
    margin-top: 0;
    padding-top: 0;
}

#md_project a {
  color: black;
  text-decoration: none;
}
#md_project a:hover {
  text-decoration: underline;
}

.task-monitor {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.task-monitor__section {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.task-monitor__title {
    font-size: 14px;
    font-weight: 600;
}

.task-monitor__cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 10px;
}

.task-monitor__card {
    border: 1px solid #d8d8d8;
    border-radius: 10px;
    padding: 12px;
    background: #fafafa;
}

.task-monitor__card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
}

.task-monitor__badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 600;
    text-transform: capitalize;
    background: #e8ecf3;
    color: #29405b;
}

.task-monitor__badge--queued {
    background: #fff4d6;
    color: #8a5b00;
}

.task-monitor__badge--in_progress {
    background: #dff4ff;
    color: #0a558c;
}

.task-monitor__badge--completed {
    background: #e3f7e8;
    color: #17633c;
}

.task-monitor__badge--failed {
    background: #fde7e9;
    color: #9f1d2d;
}

.task-monitor__source {
    font-size: 12px;
    color: #666;
    text-transform: capitalize;
}

.task-monitor__label {
    font-size: 14px;
    font-weight: 600;
    word-break: break-word;
}

.task-monitor__message,
.task-monitor__meta,
.task-monitor__empty {
    font-size: 12px;
    color: #555;
    margin-top: 6px;
    word-break: break-word;
}

.task-monitor__progress {
    margin-top: 8px;
}

.task-monitor__progress-track {
    width: 100%;
    height: 8px;
    background: #ececec;
    border-radius: 999px;
    overflow: hidden;
}

.task-monitor__progress-fill {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #4f8cff 0%, #36c0a4 100%);
}

.task-monitor__progress-text {
    margin-top: 4px;
    font-size: 12px;
    color: #555;
}
"""

MARKDOWN = """
### [Whisper-WebUI](https://github.com/jhj0517/Whsiper-WebUI)
"""


NLLB_VRAM_TABLE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }
    th {
      background-color: #f2f2f2;
    }
  </style>
</head>
<body>

<details>
  <summary>VRAM usage for each model</summary>
  <table>
    <thead>
      <tr>
        <th>Model name</th>
        <th>Required VRAM</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>nllb-200-3.3B</td>
        <td>~16GB</td>
      </tr>
      <tr>
        <td>nllb-200-1.3B</td>
        <td>~8GB</td>
      </tr>
      <tr>
        <td>nllb-200-distilled-600M</td>
        <td>~4GB</td>
      </tr>
    </tbody>
  </table>
  <p><strong>Note:</strong> Be mindful of your VRAM! The table above provides an approximate VRAM usage for each model.</p>
</details>

</body>
</html>
"""
