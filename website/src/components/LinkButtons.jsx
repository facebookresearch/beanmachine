import React from 'react';
import Link from '@docusaurus/Link';

const LinkButtons = ({githubUrl, colabUrl}) => {
  return (
    <div className="link-buttons">
      <Link to={githubUrl}>Open in GitHub</Link>
      <div></div>
      <Link to={colabUrl}>Run in Google Colab</Link>
    </div>
  );
};

export default LinkButtons;
